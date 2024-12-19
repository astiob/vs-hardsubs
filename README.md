# Dependencies

`pip install lazy`

# Usage

This script extracts a single static overlay from a sequence of frames
that have the same overlay hardsubbed on top of a dynamic background.
Pass the first and last frame on which the overlay appears and a crop
rectangle that surrounds the overlay.

`extract_hardsubs` returns the overlay itself as a pair of clips:
`(premultipliedalpha_overlay, alpha_mask)`. You can then apply it
to another clip with `std.MaskedMerge`.

`reconstruct_hardsubs` extracts the overlay and immediately applies it
to all other frames of `clip` (which defaults to the hardsubbed input
clip for easy correction of half-frame overlays that the studio
rendered in 30p on top of telecined 24p).

With either function, you will probably want to trim/rfs the returned
clips to a relevant frame range.

When dealing with real credit/lyric sequences, you will need to invoke
these functions separately for each line of lyrics and each block
of credits. It may be convenient to nest several `reconstruct_hardsubs`
calls in each other.
