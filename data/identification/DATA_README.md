# Data Identification Readme

This file described the content of the data identification folder.

### interim
This folder contains NFH examples from the automatically collected method
 (with the parse trees) with automatically detected heads 
 (Table 4 of the paper).
Each file inside that folder contain example of another rule, with all 
of the information needed.
The format of these files are:

### pickled
This directory contains the pickled version of the featured dataset


### processed
This directory contains all of the parsed and raw data for the NFH
identification part.
raw_pos and raw_neg are plain sentences (1 per line) that contain a number
which is or is not a FH depending on the file.
The rest of the files are of a similar concept but ready for training / evaluation,
automatically tokenized with a mark on where the number appears.
