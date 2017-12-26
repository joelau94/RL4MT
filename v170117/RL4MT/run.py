import sys
import os

if str(sys.argv[1]) == "preprocess":
	cmd = "python experiment/preprocess.py"
	cmd += " -c experiment/config.json"
	os.system(cmd)
elif str(sys.argv[1]) == "train":
	cmd = "python experiment/train.py"
	cmd += " -c experiment/config.json"
	cmd += " -p " + str(sys.argv[2])
	cmd += " -r models/checkpoint_model.npz" # resume model
	cmd += " -d experiment/checkpoint_status.pkl" # data status
	os.system(cmd)
elif str(sys.argv[1]) == "validate":
	cmd = "python experiment/batchvalid.py"
	cmd += " -c experiment/config.json"
	cmd += " -m models/"
	cmd += " -s " # put source here
	cmd += " -r " # put reference prefix here
	cmd += " -o " # output directory of .trans and .eval file
	# os.system(cmd)
	print("Not ready for validation.")
elif str(sys.argv[1]) == "evaluate":
	cmd = "python experiment/test.py"
	cmd += " -c experiment/config.json"
	cmd += " -m models/"
	cmd += " -s " # put source here
	cmd += " -r " # put reference prefix here
	cmd += " -v " # directory containing validation .eval file
	cmd += " -o " # output directory of .trans and .eval file
	# os.system(cmd)
	print("Not ready for test.")