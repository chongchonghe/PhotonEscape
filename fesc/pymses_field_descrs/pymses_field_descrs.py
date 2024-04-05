# File descriptor in simulation
import pymses
from pymses.sources.ramses.output import *
self.amr_field_descrs_by_file = \
    {
    "3D": {"hydro" : [ Scalar("rho", 0), Vector("vel", [1, 2, 3]), 
                       Vector("B-left", [4, 5, 6]), 
                       Vector("B-right", [7, 8, 9]), 
                       Scalar("P", 10),
                       Scalar("xHII",11), Scalar("xHeII",12), Scalar("xHeIII",13)]
           }
    }
print("Read user field descriptors by Sam Geen")
#fields = pymses.sources.ramses.output.RamsesOutput.amr_field_descrs_by_file
#hs = fields["3D"]["hydro"]
#for h in hs:
#    print h.name
