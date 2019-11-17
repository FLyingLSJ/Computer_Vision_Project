import os

init_dir = ["trainImage", "validateImage", "trainImageXML", "validateImageXML", "images", "labels"]


for i in  init_dir:
	if not os.path.exists(i):
		os.mkdir(i)
		print(f"{i} create sucessful!")
	else:
		print(f"{i} is exists")

