# OpenCV interactive calibration support

Vendored from OpenCV **4.5.4**, `apps/interactive-calibration/`:

https://github.com/opencv/opencv/tree/4.5.4/apps/interactive-calibration

License: Apache-2.0, see LICENSE. Only these support sources are built; the SDK entrypoint is ../demarcate.cpp.

Local changes:
- Clamp coverage grid indices at image edges; treat non-finite uncertainty as insufficient confidence.
- Initialize RMS; expose coverage state; use inclusive minimum and maximum view limits.
- Chessboard search on a reduced image, original-resolution subpixel refinement.
- Skip the heuristic angle rejection until a calibration exists; fix initial principal point coordinates.

SHA256 of original upstream files (before local modifications):

- `calibCommon.hpp`: `9f3ddaef2124a89739e26388d0c4424f72db506a39276156fab0b927e7fa402d`
- `calibController.cpp`: `cf62c2c632f3a2b753f99f581009ecf763e9fc487aeffbdab3efbab60b7b964e`
- `calibController.hpp`: `32f93453116422bb2e116d0f54470ef081cb29258a8f1ca078dee990811ef285`
- `frameProcessor.cpp`: `9b8992a2fa6b5447d38ab31f184b89dfe05f62364dd1c52e18f978a5e2bfe081`
- `frameProcessor.hpp`: `891f46de79c8b31eefe689c8bb38a24a34ee6456259ac0485f1c09dc430c122a`
- `rotationConverters.cpp`: `fa1668dfbb0895598ed7d776b4e05d80d320178348e6fecdfe3087d72855c8b9`
- `rotationConverters.hpp`: `6fecb3c0e10ee2655ec30b7fbf57cdef9477baaf98b728760f4fa0a87104b00e`
- `LICENSE`: `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`
