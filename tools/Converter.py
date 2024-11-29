from rpy2.robjects import numpy2ri
import rpy2.robjects as r

CONVERTER = (r.default_converter + numpy2ri.converter).context