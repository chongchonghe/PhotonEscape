# File descriptor in simulation
import pymses
from pymses.sources.ramses.output import *
self.amr_field_descrs_by_file = \
    {
    "3D": {"hydro" : [ Scalar("rho", 0), Vector("velocity", [1, 2, 3]), 
                       Scalar("pressure", 4), 
                       Scalar("metallicity", 5),
                       Scalar("xHI",6), Scalar("xHII",7), Scalar("xHeII",8), Scalar("xHeIII",9)]
           }
    }
print("Read user field descriptors for Fred's simulation")
#fields = pymses.sources.ramses.output.RamsesOutput.amr_field_descrs_by_file
#hs = fields["3D"]["hydro"]
#for h in hs:
#    print h.name
