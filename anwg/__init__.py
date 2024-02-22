__version__ = "0.0.4"

from . import waveguide
from . import dispersionCurve
from . import fields

from .waveguide import Waveguide
from .dispersionCurve import DispersionCurve
from .fields import Fields

print('ANWG v' + __version__)