import vapoursynth as vs
import numpy
from dataclasses import dataclass
from itertools import accumulate
from lazy import lazy


__all__ = 'extract_hardsubs', 'reconstruct_hardsubs'


c = vs.core


@dataclass
class LazyLeastSquares:
	op: vs.VideoNode
	ncop: vs.VideoNode

	def __post_init__(self):
		if not (0 < self.op.width == self.ncop.width and 0 < self.op.height == self.ncop.height and None != self.op.format == self.ncop.format):
			raise ValueError('Both input clips must have the same, constant format and size')

	@lazy
	def clips(self):
		op = self.op
		ncop = self.ncop

		plane_subtotal_coef = [0.0] * op.format.num_planes
		plane_subtotal_sqr_coef = [0.0] * op.format.num_planes
		plane_subtotal_val = [0.0] * op.format.num_planes
		plane_dot = [0.0] * op.format.num_planes

		for iframe, (frame, ncframe) in enumerate(zip(op.frames(close=True), ncop.frames(close=True))):
			for iplane in range(op.format.num_planes):
				plane = numpy.asarray(frame[iplane])
				ncplane = numpy.asarray(ncframe[iplane])
				input_dtype = plane.dtype
				plane_subtotal_coef[iplane] -= ncplane
				plane_subtotal_sqr_coef[iplane] += numpy.square(ncplane, dtype=numpy.float64)
				val = numpy.subtract(plane, ncplane, dtype=numpy.float64)
				plane_subtotal_val[iplane] += val
				plane_dot[iplane] -= numpy.multiply(ncplane, val, dtype=numpy.float64)
			del plane, ncplane

		alphas = []
		premultiplieds = []

		def adapt_sample_type(plane, alpha=False):
			if op.format.sample_type == vs.INTEGER:
				peak_value = (1 << op.format.bits_per_sample) - 1
				if alpha:
					plane *= peak_value
				return (numpy
					.rint(plane, plane)
					.clip(0, peak_value, plane)
					.astype(input_dtype))
			else:
				return plane

		def solve(iplane_start, iplane_stop):
			iplane_slice = slice(iplane_start, iplane_stop)
			iplane_range = range(iplane_start, iplane_stop)

			total_sqr_coef = sum(plane_subtotal_sqr_coef[iplane_slice])
			dot = sum(plane_dot[iplane_slice])

			sqr_subtotal_coef = [plane_subtotal_coef[iplane] ** 2 for iplane in iplane_range]
			subtotal_prod = [plane_subtotal_coef[iplane] * plane_subtotal_val[iplane] for iplane in iplane_range]

			rev_cumsum_sqr_subtotal_coef = list(accumulate(reversed(sqr_subtotal_coef), initial=0))
			rev_cumsum_subtotal_prod = list(accumulate(reversed(subtotal_prod), initial=0))

			denominator = op.num_frames * total_sqr_coef - rev_cumsum_sqr_subtotal_coef[-1]
			n_denominator = op.num_frames * denominator

			alphas.extend([adapt_sample_type((op.num_frames * dot - rev_cumsum_subtotal_prod[-1]) / denominator, alpha=True)] * (iplane_stop - iplane_start))

			cumsum_sqr_subtotal_coef = 0
			cumsum_subtotal_prod = 0
			for iplane in iplane_range:
				rest_sqr_subtotal_coef = cumsum_sqr_subtotal_coef + rev_cumsum_sqr_subtotal_coef[iplane_stop - 1 - iplane]
				rest_subtotal_prod = cumsum_subtotal_prod + rev_cumsum_subtotal_prod[iplane_stop - 1 - iplane]
				premultiplieds.append(adapt_sample_type(
					(total_sqr_coef * plane_subtotal_val[iplane] - dot * plane_subtotal_coef[iplane]) / denominator
					+ (rest_subtotal_prod * plane_subtotal_coef[iplane] - rest_sqr_subtotal_coef * plane_subtotal_val[iplane]) / n_denominator
				))
				if iplane < iplane_stop - 1:
					cumsum_sqr_subtotal_coef += sqr_subtotal_coef[iplane - iplane_start]
					cumsum_subtotal_prod += subtotal_prod[iplane - iplane_start]

		if op.format.subsampling_h or op.format.subsampling_w:
			solve(0, 1)
			solve(1, op.format.num_planes)
		else:
			solve(0, op.format.num_planes)

		return premultiplieds, alphas


def extract_hardsubs(op, ncop, first, last, left=0, right=0, top=0, bottom=0):
	num_frames = op.num_frames

	op = op[first:last+1].std.Crop(left=left, right=right, top=top, bottom=bottom)
	ncop = ncop[first:last+1].std.Crop(left=left, right=right, top=top, bottom=bottom)

	lstsq = LazyLeastSquares(op, ncop)

	def alphas():
		return lstsq.clips[1]

	def premultiplieds():
		return lstsq.clips[0]

	def modify_frame(array_producer):
		def callback(n, f):
			array = array_producer()
			f = f.copy()
			for iplane in range(f.format.num_planes):
				numpy.copyto(numpy.asarray(f[iplane]), array[iplane])
			return f
		return callback

	credits_alpha = op.std.BlankClip(length=1, keep=True)
	credits_premultiplied = op.std.BlankClip(length=1, keep=True)

	credits_alpha = credits_alpha.std.ModifyFrame(credits_alpha, modify_frame(alphas)) * num_frames
	credits_premultiplied = credits_premultiplied.std.ModifyFrame(credits_premultiplied, modify_frame(premultiplieds)) * num_frames

	if top:
		credits_alpha = c.std.StackVertical([credits_alpha.std.BlankClip(height=top), credits_alpha])
		credits_premultiplied = c.std.StackVertical([credits_premultiplied.std.BlankClip(height=top), credits_premultiplied])
	if bottom:
		credits_alpha = c.std.StackVertical([credits_alpha, credits_alpha.std.BlankClip(height=bottom)])
		credits_premultiplied = c.std.StackVertical([credits_premultiplied, credits_premultiplied.std.BlankClip(height=bottom)])
	if left:
		credits_alpha = c.std.StackHorizontal([credits_alpha.std.BlankClip(width=left), credits_alpha])
		credits_premultiplied = c.std.StackHorizontal([credits_premultiplied.std.BlankClip(width=left), credits_premultiplied])
	if right:
		credits_alpha = c.std.StackHorizontal([credits_alpha, credits_alpha.std.BlankClip(width=right)])
		credits_premultiplied = c.std.StackHorizontal([credits_premultiplied, credits_premultiplied.std.BlankClip(width=right)])

	credits_alpha = credits_alpha.std.SetFrameProp('_ColorRange', intval=vs.RANGE_FULL)

	return credits_premultiplied, credits_alpha


def reconstruct_hardsubs(op, ncop, reffirst, reflast, left=0, right=0, top=0, bottom=0, *, clip=None):
	credits_premultiplied, credits_alpha = extract_hardsubs(op, ncop, reffirst, reflast, left, right, top, bottom)
	return c.std.MaskedMerge(clip or op, credits_premultiplied, credits_alpha, premultiplied=True)
