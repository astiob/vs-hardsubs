import vapoursynth as vs
import ctypes
import numpy
#import time
from dataclasses import dataclass
from lazy import lazy
from vstools import depth


__all__ = 'extract_hardsubs', 'reconstruct_hardsubs'


c = vs.core
c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)


@dataclass
class LazyLeastSquares:
	op: vs.VideoNode
	ncop: vs.VideoNode
	planes_per_subsampling: list[int]

	def __post_init__(self):
		if not (0 < self.op.width == self.ncop.width and 0 < self.op.height == self.ncop.height and None != self.op.format == self.ncop.format and self.op.format.bits_per_sample == 32):
			raise ValueError('Both input clips must have the same, constant, 32-bit format and size')

	@lazy
	def clips(self):
		op = self.op
		ncop = self.ncop
		planes_per_subsampling = self.planes_per_subsampling

		a = []
		b = []
		for iplane in range(op.format.num_planes):
			if not iplane or iplane == 1 and (op.format.subsampling_h or op.format.subsampling_w):
				nplanes = planes_per_subsampling[len(a)]
				height = -(-op.height // (1 + (iplane and op.format.subsampling_h)))
				width = -(-op.width // (1 + (iplane and op.format.subsampling_w)))
				a.append(numpy.zeros((op.num_frames * nplanes, 1 + nplanes, height, width), numpy.float32))
				b.append(numpy.zeros((op.num_frames * nplanes, height, width), numpy.float32))

#		start = time.monotonic()
		for iframe, (frame, ncframe) in enumerate(zip(op.frames(), ncop.frames())):
			for iplane in range(op.format.num_planes):
				height = -(-op.height // (1 + (iplane and op.format.subsampling_h)))
				width = -(-op.width // (1 + (iplane and op.format.subsampling_w)))
				plane = ctypes.cast(frame.get_read_ptr(iplane), c_ubyte_p)
				ncplane = ctypes.cast(ncframe.get_read_ptr(iplane), c_ubyte_p)
				stride = frame.get_stride(iplane)
				ncstride = ncframe.get_stride(iplane)
				plane = numpy.ctypeslib.as_array(plane, (height, stride))
				ncplane = numpy.ctypeslib.as_array(ncplane, (height, ncstride))
				plane = plane[:, :width*frame.format.bytes_per_sample].view(numpy.float32)
				ncplane = ncplane[:, :width*ncframe.format.bytes_per_sample].view(numpy.float32)
				if op.format.subsampling_h or op.format.subsampling_w:
					if not iplane:
						a[0][iframe, 0] = -ncplane
						a[0][iframe, 1] = 1
						b[0][iframe] = plane - ncplane
					else:
						a[1][iframe * 2 + (iplane - 1), 0] = -ncplane
						a[1][iframe * 2 + (iplane - 1), iplane] = 1
						b[1][iframe * 2 + (iplane - 1)] = plane - ncplane
				else:
					a[0][iframe * op.format.num_planes + iplane, 0] = -ncplane
					a[0][iframe * op.format.num_planes + iplane, 1 + iplane] = 1
					b[0][iframe * op.format.num_planes + iplane] = plane - ncplane
#		end = time.monotonic()
#		print(end - start)

#		values = []
		alphas = []
		premultiplieds = []
		for isubsampling in range(len(a)):
			height = -(-op.height // (1 + (isubsampling and op.format.subsampling_h)))
			width = -(-op.width // (1 + (isubsampling and op.format.subsampling_w)))
#			values_planes = [numpy.zeros((height, width), numpy.float32) for i in range(planes_per_subsampling[isubsampling])]
			premultiplieds_planes = [numpy.zeros((height, width), numpy.float32) for i in range(planes_per_subsampling[isubsampling])]
			alphas_subsampling = numpy.zeros((height, width), numpy.float32)
#			values += values_planes
			premultiplieds += premultiplieds_planes
			alphas += [alphas_subsampling] * planes_per_subsampling[isubsampling]
			for y in range(height):
				for x in range(width):
					alpha, *premultiplied_planes = numpy.linalg.lstsq(a[isubsampling][..., y, x], b[isubsampling][..., y, x], rcond=None)[0]
					for i in range(planes_per_subsampling[isubsampling]):
						premultiplied = premultiplied_planes[i]
#						value = premultiplied / alpha if alpha else 0
#						values_planes[i][y, x] = value
						premultiplieds_planes[i][y, x] = premultiplied
					alphas_subsampling[y, x] = alpha
#		end = time.monotonic()
#		print(end - start)
#		return values, alphas, premultiplieds
		return premultiplieds, alphas


def extract_hardsubs(op, ncop, first, last, top=0, right=0, bottom=0, left=0):
	op = op[first:last+1].std.Crop(top=top, right=right, bottom=bottom, left=left)
	ncop = ncop[first:last+1].std.Crop(top=top, right=right, bottom=bottom, left=left)

	if op.format.subsampling_h or op.format.subsampling_w:
		planes_per_subsampling = [1, op.format.num_planes - 1]
	else:
		planes_per_subsampling = [op.format.num_planes]

	lstsq = LazyLeastSquares(op, ncop, planes_per_subsampling)

#	def values():
#		return lstsq.clips[0]

	def alphas():
		return lstsq.clips[1]

	def premultiplieds():
#		return lstsq.clips[2]
		return lstsq.clips[0]

	def modify_frame(array_producer):
		def callback(n, f):
#			start = time.monotonic()
			array = array_producer()
			f = f.copy()
			for iplane in range(f.format.num_planes):
				height = -(-f.height // (1 + (iplane and f.format.subsampling_h)))
				width = -(-f.width // (1 + (iplane and f.format.subsampling_w)))
				write = ctypes.cast(f.get_write_ptr(iplane), c_ubyte_p)
				numpy.ctypeslib.as_array(write, (height, f.get_stride(iplane)))[:, :width*f.format.bytes_per_sample].view(numpy.float32)[:] = array[iplane]
#			end = time.monotonic()
#			print(end - start)
			return f
		return callback

#	credits = op.std.BlankClip(length=1, keep=True)
	credits_alpha = op.std.BlankClip(length=1, keep=True)
	credits_premultiplied = op.std.BlankClip(length=1, keep=True)

#	credits = credits.std.ModifyFrame(credits, modify_frame(values)) * op.num_frames
	credits_alpha = credits_alpha.std.ModifyFrame(credits_alpha, modify_frame(alphas)) * op.num_frames
	credits_premultiplied = credits_premultiplied.std.ModifyFrame(credits_premultiplied, modify_frame(premultiplieds)) * op.num_frames

	if top:
#		credits = c.std.StackVertical([credits.std.BlankClip(height=top), credits])
		credits_alpha = c.std.StackVertical([credits_alpha.std.BlankClip(height=top), credits_alpha])
		credits_premultiplied = c.std.StackVertical([credits_premultiplied.std.BlankClip(height=top), credits_premultiplied])
	if bottom:
#		credits = c.std.StackVertical([credits, credits.std.BlankClip(height=bottom)])
		credits_alpha = c.std.StackVertical([credits_alpha, credits_alpha.std.BlankClip(height=bottom)])
		credits_premultiplied = c.std.StackVertical([credits_premultiplied, credits_premultiplied.std.BlankClip(height=bottom)])
	if left:
#		credits = c.std.StackHorizontal([credits.std.BlankClip(width=left), credits])
		credits_alpha = c.std.StackHorizontal([credits_alpha.std.BlankClip(width=left), credits_alpha])
		credits_premultiplied = c.std.StackHorizontal([credits_premultiplied.std.BlankClip(width=left), credits_premultiplied])
	if right:
#		credits = c.std.StackHorizontal([credits, credits.std.BlankClip(width=right)])
		credits_alpha = c.std.StackHorizontal([credits_alpha, credits_alpha.std.BlankClip(width=right)])
		credits_premultiplied = c.std.StackHorizontal([credits_premultiplied, credits_premultiplied.std.BlankClip(width=right)])

#	return credits, credits_alpha, credits_premultiplied
	return credits_premultiplied, credits_alpha


def reconstruct_hardsubs(op, ncop, reffirst, reflast, *, clip=None, top=0, right=0, bottom=0, left=0):
#	credits, credits_alpha, credits_premultiplied = extract_hardsubs(op, ncop, reffirst, reflast, top, right, bottom, left)
	credits_premultiplied, credits_alpha = extract_hardsubs(op, ncop, reffirst, reflast, top, right, bottom, left)
	return c.std.MaskedMerge(clip or op, credits_premultiplied, credits_alpha, premultiplied=True)
