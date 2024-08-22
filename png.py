__version__ = "0.0.18"

import itertools
import math
import re
import operator
import struct
import sys
import warnings
import zlib

from array import array
from functools import reduce

try:
    # cpngfilters is a Cython module: it must be compiled by
    # Cython for this import to work.
    # If this import does work, then it overrides pure-python
    # filtering functions defined later in this file (see class
    # pngfilters).
    import cpngfilters as pngfilters
except ImportError:
    pass


__all__ = ['Image', 'Reader', 'Writer', 'write_chunks', 'from_array']


# The PNG signature.
_signature = struct.pack('8B', 137, 80, 78, 71, 13, 10, 26, 10)

_adam7 = ((0, 0, 8, 8),
          (4, 0, 8, 8),
          (0, 4, 4, 8),
          (2, 0, 4, 4),
          (0, 2, 2, 4),
          (1, 0, 2, 2),
          (0, 1, 1, 2))

def group(s, n):
    # See http://www.python.org/doc/2.6/library/functions.html#zip
    return list(zip(*[iter(s)]*n))

def isarray(x):
    return isinstance(x, array)

def tostring(row):
    return row.tobytes()

def interleave_planes(ipixels, apixels, ipsize, apsize):


    itotal = len(ipixels)
    atotal = len(apixels)
    newtotal = itotal + atotal
    newpsize = ipsize + apsize
    # Set up the output buffer
    # See http://www.python.org/doc/2.4.4/lib/module-array.html#l2h-1356
    out = array(ipixels.typecode)
    # It's annoying that there is no cheap way to set the array size :-(
    out.extend(ipixels)
    out.extend(apixels)
    # Interleave in the pixel data
    for i in range(ipsize):
        out[i:newtotal:newpsize] = ipixels[i:itotal:ipsize]
    for i in range(apsize):
        out[i+ipsize:newtotal:newpsize] = apixels[i:atotal:apsize]
    return out

def check_palette(palette):
  

    # None is the default and is allowed.
    if palette is None:
        return None

    p = list(palette)
    if not (0 < len(p) <= 256):
        raise ValueError("a palette must have between 1 and 256 entries")
    seen_triple = False
    for i,t in enumerate(p):
        if len(t) not in (3,4):
            raise ValueError(
              "palette entry %d: entries must be 3- or 4-tuples." % i)
        if len(t) == 3:
            seen_triple = True
        if seen_triple and len(t) == 4:
            raise ValueError(
              "palette entry %d: all 4-tuples must precede all 3-tuples" % i)
        for x in t:
            if int(x) != x or not(0 <= x <= 255):
                raise ValueError(
                  "palette entry %d: values must be integer: 0 <= x <= 255" % i)
    return p

def check_sizes(size, width, height):
  

    if not size:
        return width, height

    if len(size) != 2:
        raise ValueError(
          "size argument should be a pair (width, height)")
    if width is not None and width != size[0]:
        raise ValueError(
          "size[0] (%r) and width (%r) should match when both are used."
            % (size[0], width))
    if height is not None and height != size[1]:
        raise ValueError(
          "size[1] (%r) and height (%r) should match when both are used."
            % (size[1], height))
    return size

def check_color(c, greyscale, which):
   

    if c is None:
        return c
    if greyscale:
        try:
            len(c)
        except TypeError:
            c = (c,)
        if len(c) != 1:
            raise ValueError("%s for greyscale must be 1-tuple" %
                which)
        if not isinteger(c[0]):
            raise ValueError(
                "%s colour for greyscale must be integer" % which)
    else:
        if not (len(c) == 3 and
                isinteger(c[0]) and
                isinteger(c[1]) and
                isinteger(c[2])):
            raise ValueError(
                "%s colour must be a triple of integers" % which)
    return c

class Error(Exception):
    def __str__(self):
        return self.__class__.__name__ + ': ' + ' '.join(self.args)

class FormatError(Error):
   

class ChunkError(FormatError):
    pass


class Writer:
   

    def __init__(self, width=None, height=None,
                 size=None,
                 greyscale=False,
                 alpha=False,
                 bitdepth=8,
                 palette=None,
                 transparent=None,
                 background=None,
                 gamma=None,
                 compression=None,
                 interlace=False,
                 bytes_per_sample=None, # deprecated
                 planes=None,
                 colormap=None,
                 maxval=None,
                 chunk_limit=2**20,
                 x_pixels_per_unit = None,
                 y_pixels_per_unit = None,
                 unit_is_meter = False):
       

        width, height = check_sizes(size, width, height)
        del size

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than zero")
        if not isinteger(width) or not isinteger(height):
            raise ValueError("width and height must be integers")
        # http://www.w3.org/TR/PNG/#7Integers-and-byte-order
        if width > 2**32-1 or height > 2**32-1:
            raise ValueError("width and height cannot exceed 2**32-1")

        if alpha and transparent is not None:
            raise ValueError(
                "transparent colour not allowed with alpha channel")

        if bytes_per_sample is not None:
            warnings.warn('please use bitdepth instead of bytes_per_sample',
                          DeprecationWarning)
            if bytes_per_sample not in (0.125, 0.25, 0.5, 1, 2):
                raise ValueError(
                    "bytes per sample must be .125, .25, .5, 1, or 2")
            bitdepth = int(8*bytes_per_sample)
        del bytes_per_sample
        if not isinteger(bitdepth) or bitdepth < 1 or 16 < bitdepth:
            raise ValueError("bitdepth (%r) must be a positive integer <= 16" %
              bitdepth)

        self.rescale = None
        palette = check_palette(palette)
        if palette:
            if bitdepth not in (1,2,4,8):
                raise ValueError("with palette, bitdepth must be 1, 2, 4, or 8")
            if transparent is not None:
                raise ValueError("transparent and palette not compatible")
            if alpha:
                raise ValueError("alpha and palette not compatible")
            if greyscale:
                raise ValueError("greyscale and palette not compatible")
        else:
            # No palette, check for sBIT chunk generation.
            if alpha or not greyscale:
                if bitdepth not in (8,16):
                    targetbitdepth = (8,16)[bitdepth > 8]
                    self.rescale = (bitdepth, targetbitdepth)
                    bitdepth = targetbitdepth
                    del targetbitdepth
            else:
                assert greyscale
                assert not alpha
                if bitdepth not in (1,2,4,8,16):
                    if bitdepth > 8:
                        targetbitdepth = 16
                    elif bitdepth == 3:
                        targetbitdepth = 4
                    else:
                        assert bitdepth in (5,6,7)
                        targetbitdepth = 8
                    self.rescale = (bitdepth, targetbitdepth)
                    bitdepth = targetbitdepth
                    del targetbitdepth

        if bitdepth < 8 and (alpha or not greyscale and not palette):
            raise ValueError(
              "bitdepth < 8 only permitted with greyscale or palette")
        if bitdepth > 8 and palette:
            raise ValueError(
                "bit depth must be 8 or less for images with palette")

        transparent = check_color(transparent, greyscale, 'transparent')
        background = check_color(background, greyscale, 'background')

        # It's important that the true boolean values (greyscale, alpha,
        # colormap, interlace) are converted to bool because Iverson's
        # convention is relied upon later on.
        self.width = width
        self.height = height
        self.transparent = transparent
        self.background = background
        self.gamma = gamma
        self.greyscale = bool(greyscale)
        self.alpha = bool(alpha)
        self.colormap = bool(palette)
        self.bitdepth = int(bitdepth)
        self.compression = compression
        self.chunk_limit = chunk_limit
        self.interlace = bool(interlace)
        self.palette = palette
        self.x_pixels_per_unit = x_pixels_per_unit
        self.y_pixels_per_unit = y_pixels_per_unit
        self.unit_is_meter = bool(unit_is_meter)

        self.color_type = 4*self.alpha + 2*(not greyscale) + 1*self.colormap
        assert self.color_type in (0,2,3,4,6)

        self.color_planes = (3,1)[self.greyscale or self.colormap]
        self.planes = self.color_planes + self.alpha
        # :todo: fix for bitdepth < 8
        self.psize = (self.bitdepth/8) * self.planes

    def make_palette(self):
        """Create the byte sequences for a `PLTE and if necessary a
        `tRNS chunk.  Returned as a pair (*p*, *t*).  *t* will be
        `None if no tRNS chunk is necessary.
        """

        p = array('B')
        t = array('B')

        for x in self.palette:
            p.extend(x[0:3])
            if len(x) > 3:
                t.append(x[3])
        p = tostring(p)
        t = tostring(t)
        if t:
            return p,t
        return p,None

    def write(self, outfile, rows):
        """Write a PNG image to the output file.  rows should be
        an iterable that yields each row in boxed row flat pixel
        format.  The rows should be the rows of the original image,
        so there should be `self.height rows of self.width *
        self.planes` values.  If interlace is specified (when
        creating the instance), then an interlaced PNG file will
        be written.  Supply the rows in the normal image order;
        the interlacing is carried out internally.
        .. note ::
          Interlacing will require the entire image to be in working
          memory.
        """

        if self.interlace:
            fmt = 'BH'[self.bitdepth > 8]
            a = array(fmt, itertools.chain(*rows))
            return self.write_array(outfile, a)

        nrows = self.write_passes(outfile, rows)
        if nrows != self.height:
            raise ValueError(
              "rows supplied (%d) does not match height (%d)" %
              (nrows, self.height))

    def write_passes(self, outfile, rows, packed=False):
        """
        Write a PNG image to the output file.
        Most users are expected to find the :meth:write or
        :meth:write_array method more convenient.
        
        The rows should be given to this method in the order that
        they appear in the output file.  For straightlaced images,
        this is the usual top to bottom ordering, but for interlaced
        images the rows should have already been interlaced before
        passing them to this function.
        rows should be an iterable that yields each row.  When
        packed is `False the rows should be in boxed row flat pixel
        format; when packed is `True each row should be a packed
        sequence of bytes.
        """

        # http://www.w3.org/TR/PNG/#5PNG-file-signature
        outfile.write(_signature)

        # http://www.w3.org/TR/PNG/#11IHDR
        write_chunk(outfile, b'IHDR',
                    struct.pack("!2I5B", self.width, self.height,
                                self.bitdepth, self.color_type,
                                0, 0, self.interlace))

        # See :chunk:order
        # http://www.w3.org/TR/PNG/#11gAMA
        if self.gamma is not None:
            write_chunk(outfile, b'gAMA',
                        struct.pack("!L", int(round(self.gamma*1e5))))

        # See :chunk:order
        # http://www.w3.org/TR/PNG/#11sBIT
        if self.rescale:
            write_chunk(outfile, b'sBIT',
                struct.pack('%dB' % self.planes,
                            *[self.rescale[0]]*self.planes))
        
        # :chunk:order: Without a palette (PLTE chunk), ordering is
        # relatively relaxed.  With one, gAMA chunk must precede PLTE
        # chunk which must precede tRNS and bKGD.
        # See http://www.w3.org/TR/PNG/#5ChunkOrdering
        if self.palette:
            p,t = self.make_palette()
            write_chunk(outfile, b'PLTE', p)
            if t:
                # tRNS chunk is optional. Only needed if palette entries
                # have alpha.
                write_chunk(outfile, b'tRNS', t)

        # http://www.w3.org/TR/PNG/#11tRNS
        if self.transparent is not None:
            if self.greyscale:
                write_chunk(outfile, b'tRNS',
                            struct.pack("!1H", *self.transparent))
            else:
                write_chunk(outfile, b'tRNS',
                            struct.pack("!3H", *self.transparent))

        # http://www.w3.org/TR/PNG/#11bKGD
        if self.background is not None:
            if self.greyscale:
                write_chunk(outfile, b'bKGD',
                            struct.pack("!1H", *self.background))
            else:
                write_chunk(outfile, b'bKGD',
                            struct.pack("!3H", *self.background))

        # http://www.w3.org/TR/PNG/#11pHYs
        if self.x_pixels_per_unit is not None and self.y_pixels_per_unit is not None:
            tup = (self.x_pixels_per_unit, self.y_pixels_per_unit, int(self.unit_is_meter))
            write_chunk(outfile, b'pHYs', struct.pack("!LLB",*tup))

        # http://www.w3.org/TR/PNG/#11IDAT
        if self.compression is not None:
            compressor = zlib.compressobj(self.compression)
        else:
            compressor = zlib.compressobj()

        # Choose an extend function based on the bitdepth.  The extend
        # function packs/decomposes the pixel values into bytes and
        # stuffs them onto the data array.
        data = array('B')
        if self.bitdepth == 8 or packed:
            extend = data.extend
        elif self.bitdepth == 16:
            # Decompose into bytes
            def extend(sl):
                fmt = '!%dH' % len(sl)
                data.extend(array('B', struct.pack(fmt, *sl)))
        else:
            # Pack into bytes
            assert self.bitdepth < 8
            # samples per byte
            spb = int(8/self.bitdepth)
            def extend(sl):
                a = array('B', sl)
                # Adding padding bytes so we can group into a whole
                # number of spb-tuples.
                l = float(len(a))
                extra = math.ceil(l / float(spb))*spb - l
                a.extend([0]*int(extra))
                # Pack into bytes
                l = group(a, spb)
                l = [reduce(lambda x,y:
                                           (x << self.bitdepth) + y, e) for e in l]
                data.extend(l)
        if self.rescale:
            oldextend = extend
            factor = \
              float(2**self.rescale[1]-1) / float(2**self.rescale[0]-1)
            def extend(sl):
                oldextend([int(round(factor*x)) for x in sl])

        # Build the first row, testing mostly to see if we need to
        # changed the extend function to cope with NumPy integer types
        # (they cause our ordinary definition of extend to fail, so we
        # wrap it).  See
        # http://code.google.com/p/pypng/issues/detail?id=44
        enumrows = enumerate(rows)
        del rows

        # First row's filter type.
        data.append(0)
        # :todo: Certain exceptions in the call to `.next() or the
        # following try would indicate no row data supplied.
        # Should catch.
        i,row = next(enumrows)
        try:
            # If this fails...
            extend(row)
        except:
            # ... try a version that converts the values to int first.
            # Not only does this work for the (slightly broken) NumPy
            # types, there are probably lots of other, unknown, "nearly"
            # int types it works for.
            def wrapmapint(f):
                return lambda sl: f([int(x) for x in sl])
            extend = wrapmapint(extend)
            del wrapmapint
            extend(row)

        for i,row in enumrows:
            # Add "None" filter type.  Currently, it's essential that
            # this filter type be used for every scanline as we do not
            # mark the first row of a reduced pass image; that means we
            # could accidentally compute the wrong filtered scanline if
            # we used "up", "average", or "paeth" on such a line.
            data.append(0)
            extend(row)
            if len(data) > self.chunk_limit:
                compressed = compressor.compress(tostring(data))
                if len(compressed):
                    write_chunk(outfile, b'IDAT', compressed)
                # Because of our very witty definition of `extend,
                # above, we must re-use the same `data object.  Hence
                # we use `del to empty this one, rather than create a
                # fresh one (which would be my natural FP instinct).
                del data[:]
        if len(data):
            compressed = compressor.compress(tostring(data))
        else:
            compressed = b''
        flushed = compressor.flush()
        if len(compressed) or len(flushed):
            write_chunk(outfile, b'IDAT', compressed + flushed)
        # http://www.w3.org/TR/PNG/#11IEND
        write_chunk(outfile, b'IEND')
        return i+1

    def write_array(self, outfile, pixels):
        """
        Write an array in flat row flat pixel format as a PNG file on
        the output file.  See also :meth:write method.
        """

        if self.interlace:
            self.write_passes(outfile, self.array_scanlines_interlace(pixels))
        else:
            self.write_passes(outfile, self.array_scanlines(pixels))

    def write_packed(self, outfile, rows):
        """
        Write PNG file to outfile.  The pixel data comes from rows
        which should be in boxed row packed format.  Each row should be
        a sequence of packed bytes.
        Technically, this method does work for interlaced images but it
        is best avoided.  For interlaced images, the rows should be
        presented in the order that they appear in the file.
        This method should not be used when the source image bit depth
        is not one naturally supported by PNG; the bit depth should be
        1, 2, 4, 8, or 16.
        """

        if self.rescale:
            raise Error("write_packed method not suitable for bit depth %d" %
              self.rescale[0])
        return self.write_passes(outfile, rows, packed=True)

    def convert_pnm(self, infile, outfile):
        """
        Convert a PNM file containing raw pixel data into a PNG file
        with the parameters set in the writer object.  Works for
        (binary) PGM, PPM, and PAM formats.
        """

        if self.interlace:
            pixels = array('B')
            pixels.fromfile(infile,
                            (self.bitdepth/8) * self.color_planes *
                            self.width * self.height)
            self.write_passes(outfile, self.array_scanlines_interlace(pixels))
        else:
            self.write_passes(outfile, self.file_scanlines(infile))

    def convert_ppm_and_pgm(self, ppmfile, pgmfile, outfile):
        """
        Convert a PPM and PGM file containing raw pixel data into a
        PNG outfile with the parameters set in the writer object.
        """
        pixels = array('B')
        pixels.fromfile(ppmfile,
                        (self.bitdepth/8) * self.color_planes *
                        self.width * self.height)
        apixels = array('B')
        apixels.fromfile(pgmfile,
                         (self.bitdepth/8) *
                         self.width * self.height)
        pixels = interleave_planes(pixels, apixels,
                                   (self.bitdepth/8) * self.color_planes,
                                   (self.bitdepth/8))
        if self.interlace:
            self.write_passes(outfile, self.array_scanlines_interlace(pixels))
        else:
            self.write_passes(outfile, self.array_scanlines(pixels))

    def file_scanlines(self, infile):
        """
        Generates boxed rows in flat pixel format, from the input file
        infile.  It assumes that the input file is in a "Netpbm-like"
        binary format, and is positioned at the beginning of the first
        pixel.  The number of pixels to read is taken from the image
        dimensions (width, height, planes) and the number of bytes
        per value is implied by the image bitdepth.
        """

        # Values per row
        vpr = self.width * self.planes
        row_bytes = vpr
        if self.bitdepth > 8:
            assert self.bitdepth == 16
            row_bytes *= 2
            fmt = '>%dH' % vpr
            def line():
                return array('H', struct.unpack(fmt, infile.read(row_bytes)))
        else:
            def line():
                scanline = array('B', infile.read(row_bytes))
                return scanline
        for y in range(self.height):
            yield line()

    def array_scanlines(self, pixels):
        """
        Generates boxed rows (flat pixels) from flat rows (flat pixels)
        in an array.
        """

        # Values per row
        vpr = self.width * self.planes
        stop = 0
        for y in range(self.height):
            start = stop
            stop = start + vpr
            yield pixels[start:stop]

    def array_scanlines_interlace(self, pixels):
        """
        Generator for interlaced scanlines from an array.  pixels is
        the full source image in flat row flat pixel format.  The
        generator yields each scanline of the reduced passes in turn, in
        boxed row flat pixel format.
        """

        # http://www.w3.org/TR/PNG/#8InterlaceMethods
        # Array type.
        fmt = 'BH'[self.bitdepth > 8]
        # Value per row
        vpr = self.width * self.planes
        for xstart, ystart, xstep, ystep in _adam7:
            if xstart >= self.width:
                continue
            # Pixels per row (of reduced image)
            ppr = int(math.ceil((self.width-xstart)/float(xstep)))
            # number of values in reduced image row.
            row_len = ppr*self.planes
            for y in range(ystart, self.height, ystep):
                if xstep == 1:
                    offset = y * vpr
                    yield pixels[offset:offset+vpr]
                else:
                    row = array(fmt)
                    # There's no easier way to set the length of an array
                    row.extend(pixels[0:row_len])
                    offset = y * vpr + xstart * self.planes
                    end_offset = (y+1) * vpr
                    skip = self.planes * xstep
                    for i in range(self.planes):
                        row[i::self.planes] = \
                            pixels[offset+i:end_offset:skip]
                    yield row

def write_chunk(outfile, tag, data=b''):
    """
    Write a PNG chunk to the output file, including length and
    checksum.
    """

    # http://www.w3.org/TR/PNG/#5Chunk-layout
    outfile.write(struct.pack("!I", len(data)))
    outfile.write(tag)
    outfile.write(data)
    checksum = zlib.crc32(tag)
    checksum = zlib.crc32(data, checksum)
    checksum &= 2**32-1
    outfile.write(struct.pack("!I", checksum))

def write_chunks(out, chunks):
    """Create a PNG file by writing out the chunks."""

    out.write(_signature)
    for chunk in chunks:
        write_chunk(out, *chunk)

def filter_scanline(type, line, fo, prev=None):
    """Apply a scanline filter to a scanline.  type specifies the
    filter type (0 to 4); line specifies the current (unfiltered)
    scanline as a sequence of bytes; prev specifies the previous
    (unfiltered) scanline as a sequence of bytes. fo specifies the
    filter offset; normally this is size of a pixel in bytes (the number
    of bytes per sample times the number of channels), but when this is
    < 1 (for bit depths < 8) then the filter offset is 1.
    """

    assert 0 <= type < 5

    # The output array.  Which, pathetically, we extend one-byte at a
    # time (fortunately this is linear).
    out = array('B', [type])

    def sub():
        ai = -fo
        for x in line:
            if ai >= 0:
                x = (x - line[ai]) & 0xff
            out.append(x)
            ai += 1
    def up():
        for i,x in enumerate(line):
            x = (x - prev[i]) & 0xff
            out.append(x)
    def average():
        ai = -fo
        for i,x in enumerate(line):
            if ai >= 0:
                x = (x - ((line[ai] + prev[i]) >> 1)) & 0xff
            else:
                x = (x - (prev[i] >> 1)) & 0xff
            out.append(x)
            ai += 1
    def paeth():
        # http://www.w3.org/TR/PNG/#9Filter-type-4-Paeth
        ai = -fo # also used for ci
        for i,x in enumerate(line):
            a = 0
            b = prev[i]
            c = 0

            if ai >= 0:
                a = line[ai]
                c = prev[ai]
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            if pa <= pb and pa <= pc:
                Pr = a
            elif pb <= pc:
                Pr = b
            else:
                Pr = c

            x = (x - Pr) & 0xff
            out.append(x)
            ai += 1

    if not prev:
        # We're on the first line.  Some of the filters can be reduced
        # to simpler cases which makes handling the line "off the top"
        # of the image simpler.  "up" becomes "none"; "paeth" becomes
        # "left" (non-trivial, but true). "average" needs to be handled
        # specially.
        if type == 2: # "up"
            type = 0
        elif type == 3:
            prev = [0]*len(line)
        elif type == 4: # "paeth"
            type = 1
    if type == 0:
        out.extend(line)
    elif type == 1:
        sub()
    elif type == 2:
        up()
    elif type == 3:
        average()
    else: # type == 4
        paeth()
    return out


# Regex for decoding mode string
RegexModeDecode = re.compile("(LA?|RGBA?);?([0-9]*)", flags=re.IGNORECASE)

def from_array(a, mode=None, info={}):
    