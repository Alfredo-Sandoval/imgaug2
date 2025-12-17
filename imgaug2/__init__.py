"""Imports for package imgaug2."""

# this contains some deprecated classes/functions pointing to the new
# classes/functions, hence always place the other imports below this so that
# the deprecated stuff gets overwritten as much as possible
from imgaug2.imgaug import *  # pylint: disable=redefined-builtin

import imgaug2.augmentables as augmentables
from imgaug2.augmentables import *
import imgaug2.augmenters as augmenters
import imgaug2.parameters as parameters
import imgaug2.dtypes as dtypes
import imgaug2.data as data

__version__ = "0.5.0"
