A bunch of convenience functions for reading Horizon:AGN data on infinity

uses/requires some other stuff from me

https://github.com/jlewis47/GREMLIN (python objects that handle RAMSES simulation information like header or info data)
https://github.com/jlewis47/fortran_pyutils (convenience stuff for reading the unformatted fortran binary files)

association.py contains functions for looking up stellar data using the hid or gid
tree_reader.py contains functions for parsing the merger tree of galaxies and halos (soon)
Note that implementation uses pre-processed lookup files to jump through the tree file and avoid reading it all. These are linked to in the functions on infinity.
The same .py file also contains the function used to generate these files.
