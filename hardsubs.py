import vapoursynth as vs
import numpy
from dataclasses import dataclass
from itertools import accumulate
from lazy import lazy


__all__ = 'extract_hardsubs', 'reconstruct_hardsubs'


@dataclass
class LazyLeastSquares:
	"""
	Wrapper that allows two ``vapoursynth.VideoFrame``s
	to be produced together in a single deferred/lazy computation.
	"""

	op: vs.VideoNode
	ncop: vs.VideoNode
	left: int
	right: int
	top: int
	bottom: int

	@lazy
	def frames(self):
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

		alpha_planes = []
		premultiplied_planes = []

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
			"""
			Solve the ordinary least-squares problem for a number of co-sited planes.

			We assume that the authoring studio used a single, static overlay
			and alpha-blended (hardsubbed) it onto a sequence of moving frames.
			We have the clean frames and the hardsubbed ones.
			For each individual sample, we know that::

				hardsubbed_sample[frame 0] = overlay*alpha + clean_sample[frame 0]*(1-alpha)
				hardsubbed_sample[frame 1] = overlay*alpha + clean_sample[frame 1]*(1-alpha)
				hardsubbed_sample[frame 2] = overlay*alpha + clean_sample[frame 2]*(1-alpha)
				...

			So we run a least-squares linear regression for each sample
			to find the ``alpha`` and the ``overlay*alpha``, and we're done.

			Least-squares problems are usually written as::

				a0 * x + b0 * y = c0
				a1 * x + b1 * y = c1
				a2 * x + b2 * y = c2
				...

			Rearrange our equations to match this form,
			setting ``b`` to a constant ``1`` for convenience::

				hardsubbed = overlay*alpha + clean*(1-alpha)
				hardsubbed = overlay*alpha + clean - clean*alpha
				(-clean) * alpha + 1 * (overlay*alpha) = (hardsubbed-clean)

			thus::

				a = -clean     x = alpha
				b = 1          y = overlay*alpha
				c = hardsubbed-clean

			The video clips may have multiple planes (YUV, RGB).
			We could compute the overlay's alpha mask separately for each plane,
			but presumably, the authoring studio had a single alpha mask for the whole clip,
			so we try to restore the single mask by combining data from all planes.

			However, subsampled chroma inevitably uses a subsampled version of the alpha mask.
			To avoid any additional lossy resampling of the alpha mask,
			we compute one mask for all the non-subsampled planes
			and a separate mask for all the subsampled planes,
			assuming that the subsampled planes are co-sited with each other.
			Each ``solve`` call corresponds to one of these groups of planes.

			When solving for multiple planes,
			the combined least-squares problem looks like this::

				a[0]    * alpha + overlay_planeA*alpha                 = c[0]
				a[1]    * alpha + overlay_planeA*alpha                 = c[1]
				a[2]    * alpha + overlay_planeA*alpha                 = c[2]
				...
				a[N+0]  * alpha         + overlay_planeB*alpha         = c[N+0]
				a[N+1]  * alpha         + overlay_planeB*alpha         = c[N+1]
				a[N+2]  * alpha         + overlay_planeB*alpha         = c[N+2]
				...
				a[2N+0] * alpha                 + overlay_planeC*alpha = c[2N+0]
				a[2N+1] * alpha                 + overlay_planeC*alpha = c[2N+1]
				a[2N+2] * alpha                 + overlay_planeC*alpha = c[2N+2]
				...

			and the solution consists of the following values::

				alpha
				overlay_planeA*alpha
				overlay_planeB*alpha
				overlay_planeC*alpha
			"""

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

			alpha_planes.extend([adapt_sample_type((op.num_frames * dot - rev_cumsum_subtotal_prod[-1]) / denominator, alpha=True)] * (iplane_stop - iplane_start))

			cumsum_sqr_subtotal_coef = 0
			cumsum_subtotal_prod = 0
			for iplane in iplane_range:
				rest_sqr_subtotal_coef = cumsum_sqr_subtotal_coef + rev_cumsum_sqr_subtotal_coef[iplane_stop - 1 - iplane]
				rest_subtotal_prod = cumsum_subtotal_prod + rev_cumsum_subtotal_prod[iplane_stop - 1 - iplane]
				premultiplied_planes.append(adapt_sample_type(
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

		frames = []
		for array in premultiplied_planes, alpha_planes:
			frame = vs.core.create_video_frame(op.format, self.left + op.width + self.right, self.top + op.height + self.bottom)
			frames.append(frame)
			for iplane in range(op.format.num_planes):
				left = self.left >> (iplane and op.format.subsampling_w)
				right = self.right >> (iplane and op.format.subsampling_w)
				top = self.top >> (iplane and op.format.subsampling_h)
				bottom = self.bottom >> (iplane and op.format.subsampling_h)

				background = 1 << (op.format.bits_per_sample - 1) if (
					array is premultiplied_planes
					and iplane
					and op.format.color_family == vs.YUV
					and op.format.sample_type == vs.INTEGER
				) else 0

				outarray = numpy.asarray(frame[iplane])
				outarray[:top] = background
				outarray[top:-bottom, :left] = background
				numpy.copyto(outarray[top:-bottom, left:-right], array[iplane])
				outarray[top:-bottom, -right:] = background
				outarray[-bottom:] = background

		return frames


def extract_hardsubs(op, ncop, first, last, left=0, right=0, top=0, bottom=0):
	"""
	Extract a single static overlay alpha-blended (hardsubbed) onto a sequence of moving frames,
	given the clean frames and the hardsubbed ones.

	For each individual sample, it is assumed that::

		hardsubbed_sample[frame 0] = overlay*alpha + clean_sample[frame 0]*(1-alpha)
		hardsubbed_sample[frame 1] = overlay*alpha + clean_sample[frame 1]*(1-alpha)
		hardsubbed_sample[frame 2] = overlay*alpha + clean_sample[frame 2]*(1-alpha)
		...

	A least-squares linear regression is run for each sample to find the ``alpha`` and the ``overlay*alpha``.
	For details, see ``solve`` inside ``LazyLeastSquares.frames``.

	When one overlay is visible on screen and another overlay appears before the first one disappears
	(e.g. song lyrics and credits during an opening), these overlays should be extracted separately.
	Draw a rough rectangle that fully encloses each overlay (such that the different overlays' rectangles
	don't overlap) and pass it in the ``left``, ``right``, ``top``, ``bottom`` parameters as for ``std.Crop``.

	:param op:       Hardsubbed clip (e.g. opening).
	:param ncop:     Corresponding clean clip (e.g. creditless opening).
	:param first:    First frame index on which the hardsubbed overlay appears.
	:param last:     Last frame index on which the hardsubbed overlay appears.
	:param left:     Number of ignored pixels on the left side of the frame.
	:param right:    Number of ignored pixels on the right side of the frame.
	:param top:      Number of ignored pixels on the top side of the frame.
	:param bottom:   Number of ignored pixels on the bottom side of the frame.

	:return:         A tuple of two clips: the extracted overlay's alpha-premultiplied colors
	                 and the extracted overlay's alpha mask (defined for each plane).
	                 These clips are ready to be passed to ``std.MaskedMerge(premultiplied=True)``.
	"""

	if not (0 < op.width == ncop.width and 0 < op.height == ncop.height and None != op.format == ncop.format):
		raise ValueError('Both input clips must have the same, constant format and size')

	# Rely on Crop to validate the crop parameters against the clip's subsampling scheme
	lstsq = LazyLeastSquares(
		op[first:last+1].std.Crop(left=left, right=right, top=top, bottom=bottom),
		ncop[first:last+1].std.Crop(left=left, right=right, top=top, bottom=bottom),
		left=left, right=right, top=top, bottom=bottom,
	)

	credits_alpha = op.std.ModifyFrame(op, lambda n, f: lstsq.frames[1])
	credits_premultiplied = op.std.ModifyFrame(op, lambda n, f: lstsq.frames[0])

	prop_names = [
		'_ChromaLocation',
		'_Primaries',
		'_Matrix',
		'_Transfer',
		'_AbsoluteTime',
		'_DurationNum',
		'_DurationDen',
		'_SARNum',
		'_SARDen',
	]
	credits_alpha = (credits_alpha
		.std.CopyFrameProps(op, prop_names)
		.std.SetFrameProp('_ColorRange', intval=vs.RANGE_FULL)
		.std.SetFieldBased(vs.FIELD_PROGRESSIVE))

	prop_names.append('_ColorRange')
	credits_premultiplied = (credits_premultiplied
		.std.CopyFrameProps(op, prop_names)
		.std.SetFieldBased(vs.FIELD_PROGRESSIVE))

	return credits_premultiplied, credits_alpha


def reconstruct_hardsubs(op, ncop, reffirst, reflast, left=0, right=0, top=0, bottom=0, *, clip=None):
	credits_premultiplied, credits_alpha = extract_hardsubs(op, ncop, reffirst, reflast, left, right, top, bottom)
	return vs.core.std.MaskedMerge(clip or op, credits_premultiplied, credits_alpha, premultiplied=True)
