#!/bin/bash

configure_imagemagick() {
  if "$PYTHON" -c 'from wand.image import Image' >/dev/null 2>&1; then
    return
  fi

  local experiment_root candidate
  experiment_root="$(cd "$REPO_ROOT/../.." && pwd)"
  candidate="${IMAGEMAGICK_ROOT:-${MAGICK_HOME:-$experiment_root/envs/imagemagick}}"
  if [ ! -f "$candidate/lib/libMagickWand-7.Q16HDRI.so" ]; then
    echo "MagickWand is unavailable; set IMAGEMAGICK_ROOT to a valid private ImageMagick prefix" >&2
    return 1
  fi

  export MAGICK_HOME="$candidate"
  export LD_LIBRARY_PATH="$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="$MAGICK_HOME/bin:$PATH"
  export WAND_MAGICK_LIBRARY_SUFFIX="-7.Q16HDRI"
  export LD_PRELOAD="$MAGICK_HOME/lib/libMagickCore-7.Q16HDRI.so:$MAGICK_HOME/lib/libMagickWand-7.Q16HDRI.so${LD_PRELOAD:+:$LD_PRELOAD}"
  "$PYTHON" -c 'from wand.image import Image'
}
