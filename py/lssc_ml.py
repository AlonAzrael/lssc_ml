
from py_utils.Utils_Base import *
from multiprocessing import Pool

import cv2
import numpy as np

from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.tree import ExtraTreeClassifier as ExTreeClf
from sklearn.ensemble import ExtraTreesClassifier as ExTreesClf


# for debug
CONFIG = {
	"sample_shape_path":"../image_dataset/sample_shape/",
	"std_shape_path":"../image_dataset/std_shape/",
}
DIR_SAMPLE_SHAPE = Dir(path=CONFIG["sample_shape_path"])
DIR_STD_SHAPE = Dir(path=CONFIG["std_shape_path"])


"""
LSSC as core
==============================================
"""

class LSSC(object):

	def __init__(self, filepath=None, image_string=None, is_bounding=True, n_lssc_rows=50, n_lssc_columns=50):
		self.read_image_as_mat(filepath, image_string, is_bounding, n_lssc_rows, n_lssc_columns)

		self.is_lssc_flag = True

	def read_image_as_mat(self, filepath=None, image_string=None, is_bounding=True, n_lssc_rows=50, n_lssc_columns=50):

		if image_string is not None:
			nparr = np.fromstring(image_string, np.uint8)
			image = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_GRAYSCALE)
		else:
			# image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
			image = cv2.imread(filepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)

		thresh, image_binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		
		if is_bounding:
			image_binary = self.bounding_image(image_binary)
		image_out = self.resize_by_grid(image_binary, n_row=n_lssc_rows, n_column=n_lssc_columns, non_zero_val=1)

		image_out_show = np.where(image_out > 0, image_out, 255)
		# show_image(image_out_show)
		cv2.imwrite(DIR_STD_SHAPE.path_join("{size}*{size}.grid_resize.jpg".format(size=n_lssc_rows)), image_out_show)

		image_resized = cv2.resize(image_binary, (n_lssc_rows, n_lssc_columns)) 
		# show_image(image_resized)
		cv2.imwrite(DIR_STD_SHAPE.path_join("{size}*{size}.resize.jpg".format(size=n_lssc_rows)), image_resized)


		# print image_out
		self.shape_mat = image_out
		self.shape_pixels = self.find_shape_pixels(self.shape_mat)
		self.shape_pixels_kd_tree = KDTree(self.shape_pixels, leaf_size=5)

	def get_shape_mat(self):
		return self._shape_mat

	def get_shape_pixels(self):
		return self._shape_pixels

	def find_shape_pixels(self, shape_mat, non_zero_val=1):
		return np.asarray(zip(*np.where(shape_mat >= non_zero_val)))

	"""
	bounding image
	"""

	def bounding_image(self, image):
		bounding_rect = self.find_bounding_rect(image)
		image_bounding = self.crop_image(image, bounding_rect)

		return image_bounding

	def find_bounding_rect(self, image):
		row_indices, column_indices = np.where(image < 128)
		bounding_rect = dict(
			top = np.amin(row_indices),
			bottom = np.amax(row_indices),
			left = np.amin(column_indices),
			right = np.amax(column_indices))
		
		return bounding_rect

	def crop_image(self, image, bounding_rect):
		return image[bounding_rect["top"]:bounding_rect["bottom"]+1,bounding_rect["left"]:bounding_rect["right"]+1]

	def is_pixel_isolated(self, pixel_coor, image_binary):
		"""better bounding rect"""
		pass

	"""
	resize image by grid
	"""

	def resize_by_grid(self, image, n_row=30, n_column=30, non_zero_val=1):
		image_resized = np.zeros(shape=(n_row,n_column), dtype=np.uint8)

		cur_n_row, cur_n_column = image.shape
		cell_height = 1.0*cur_n_row/n_row
		cell_width = 1.0*cur_n_column/n_column

		for index in self.find_black_pixels(image):
			cur_row, cur_column = index
			image_resized[int(cur_row/cell_height)][int(cur_column/cell_width)] = non_zero_val

		# print image_resized
		return image_resized

	def find_black_pixels(self, image):
		row_indices, column_indices = np.where(image < 128)
		black_pixel_indices = np.asarray(zip(*[row_indices, column_indices]))
		return black_pixel_indices

	"""
	calc image sim
	"""

	def calc_sim(self, shape_pixels):
		"""

		"""
		if getattr(shape_pixels, "is_lssc_flag", False):
			lssc = shape_pixels
			shape_pixels = lssc.shape_pixels

		kd_tree = self.shape_pixels_kd_tree
		
		# calc dist_arr of shape_pixels
		dist_arr, index_arr = kd_tree.query(shape_pixels, k=1, return_distance=True, dualtree=False)
		dist_arr = dist_arr.flatten()
		dist_arr = np.where(dist_arr > 0, dist_arr, 0.5)

		# how to make dist_arr sensitive to shape change
		# sim = np.sum( (1.0/(dist_arr**1)) )**1 # dont use this unstable deveil
		sim = np.sum( dist_arr )**0.5
		return sim

def create_lssc_n_jobs(dirx, n_jobs=4, is_bounding=True):
	n_process = n_jobs
	filepath_arr = [fn for fn in dirx.filenames() if not fn.startswith(".")]
	task_params = [dict(filepath_arr=fp_arr, is_bounding=is_bounding, dirx=dirx) for fp_arr in split_seq(filepath_arr, n_process)]

	pool = Pool()
	task_results = pool.map(create_lssc_process, task_params)

	result = {}
	[result.update(tr) for tr in task_results]
	return result

def create_lssc_process(params):
	filepath_arr = params["filepath_arr"]
	is_bounding = params["is_bounding"]
	dirx = params["dirx"]
	
	result = {}
	for fn in filepath_arr:
		fp = dirx.path_join(fn)
		result[fn] = LSSC(fp, is_bounding=is_bounding)

	return result

def gen_sample_vec_n_jobs(sample_lssc_dict, std_lssc_arr, n_jobs=4):
	task_params = [dict(std_lssc_arr=std_lssc_arr, sample_lssc_dict=dict(part_items)) for part_items in split_seq(sample_lssc_dict.items(), n_jobs)]
	pool = Pool()
	task_results = pool.map(gen_sample_vec_process, task_params)

	result = {}
	[result.update(tr) for tr in task_results]
	return result

def gen_sample_vec_process(params):
	std_lssc_arr = params["std_lssc_arr"]
	sample_lssc_dict = params["sample_lssc_dict"]
	
	result = {}
	for fn,sample_lssc in sample_lssc_dict.items():
		vec = []
		for std_lssc in std_lssc_arr:
			sim = sample_lssc.calc_sim(std_lssc)
			vec.append(sim)
		result[fn] = vec

	return result


"""
use LSSC to transform image to vec
==============================================
"""

def load_lssc_process(params):
	filepath_arr = params["filepath_arr"]
	# is_bounding = params["is_bounding"]
	# n_lssc_rows = params["n_lssc_rows"]
	# n_lssc_columns = params["n_lssc_columns"]
	lssc_params = params["lssc_params"]
	
	lssc_arr = []
	for fp in filepath_arr:
		lssc = LSSC(fp, **lssc_params)
		lssc_arr.append(lssc)

	return {"lssc_arr": lssc_arr, "task_index": params["task_index"]}

def load_lssc_arr(filepath_arr, lssc_params={}, n_jobs=4):
	lssc_arr = None
	if n_jobs>1:
		task_params = [dict(filepath_arr=fp_arr, lssc_params=lssc_params, task_index=i) for i,fp_arr in enumerate(split_seq(filepath_arr, n_jobs))]

		pool = Pool()
		task_results = pool.map(load_lssc_process, task_params)

		result = [0]*n_jobs
		for tr in task_results:
			task_index = tr["task_index"]
			result[task_index] = tr["lssc_arr"]
		
		lssc_arr = []
		for x in result:
			lssc_arr.extend(x)

	elif n_jobs == 1:
		lssc_arr = load_lssc_process(dict(filepath_arr=filepath_arr,  lssc_params=lssc_params, task_index=1))["lssc_arr"]

	else:
		raise Exception("fuck")

	return lssc_arr

def convert_lssc_to_vec_process(params):
	std_lssc_arr = params["std_lssc_arr"]
	lssc_arr = params["lssc_arr"]
	
	lssc_vec_arr = []
	for lssc in lssc_arr:
		vec = []
		for std_lssc in std_lssc_arr:
			sim = lssc.calc_sim(std_lssc)
			vec.append(sim)
		lssc_vec_arr.append(vec)

	return dict(lssc_vec_arr=lssc_vec_arr, task_index=params["task_index"])


def convert_lssc_to_vec_arr(lssc_arr, std_lssc_arr, n_jobs=4):
	lssc_vec_arr = None
	if n_jobs>1:
		task_params = [dict(lssc_arr=sub_lssc_arr, std_lssc_arr=std_lssc_arr, task_index=i) for i,sub_lssc_arr in enumerate(split_seq(lssc_arr, n_jobs))]

		pool = Pool()
		task_results = pool.map(convert_lssc_to_vec_process, task_params)

		result = [0]*n_jobs
		for tr in task_results:
			result[tr["task_index"]] = tr["lssc_vec_arr"]
		
		lssc_vec_arr = []
		for x in result:
			lssc_vec_arr.extend(x)

	elif n_jobs == 1:
		lssc_vec_arr = convert_lssc_to_vec_process(dict(lssc_arr=lssc_arr, std_lssc_arr=std_lssc_arr, task_index=1))["lssc_vec_arr"]

	else:
		raise Exception("fuck")

	return lssc_vec_arr


class LSSC_ML():

	def __init__(self, n_jobs=1):
		# self.lssc_params = lssc_params
		self.n_jobs = n_jobs

	def fit_transform(self, sample_filepath_arr, std_filepath_arr=None):
		lssc_arr = load_lssc_arr(sample_filepath_arr, lssc_params=self.lssc_params, n_jobs=self.n_jobs)
		if std_filepath_arr is None:
			std_lssc_arr = lssc_arr
		else:
			lssc_params = self.lssc_params
			lssc_params["is_bounding"] = False
			std_lssc_arr = load_lssc_arr(std_filepath_arr, lssc_params=lssc_params, n_jobs=self.n_jobs)

		self.lssc_arr = lssc_arr
		self.std_lssc_arr = std_lssc_arr

	def fit(self, std_filepath_arr, lssc_params={}):
		# lssc_params["is_bounding"] = False
		self.std_lssc_arr = load_lssc_arr(std_filepath_arr, lssc_params=lssc_params, n_jobs=self.n_jobs)

	def transform(self, sample_filepath_arr=None,  lssc_params={}):
		if sample_filepath_arr is None:
			sample_lssc_arr = self.std_lssc_arr
		else:
			sample_lssc_arr = load_lssc_arr(sample_filepath_arr, lssc_params=lssc_params, n_jobs=self.n_jobs)

		lssc_vec_arr = convert_lssc_to_vec_arr(sample_lssc_arr, self.std_lssc_arr, n_jobs=self.n_jobs)
		return np.asarray(lssc_vec_arr)

def exp_score(X, Y):
	model = KNC(n_neighbors=3, weights="distance", algorithm="brute")
	model.fit(X, Y)

	pY = []
	for i,_ in enumerate(X):
		tx, ty = X[i], Y[i]
		dX = np.delete(X, [i], axis=0)
		dY = np.delete(Y, [i], axis=0)
		# print dY
		# print tx, ty
		model.fit(dX, dY)

		predict_y = model.predict([tx])
		print tx, ty, predict_y[0] == ty


"""
API
==============================================
"""

def get_dir_filepath_arr(dd):
	return [dd.path_join(fn) for fn in dd.filenames() if not fn.startswith(".")]

def get_dir_fp_and_label_arr(dd):
	sample_filepath_arr = get_dir_filepath_arr(dd)
	sample_labels = [int(os.path.basename(fp).split(".")[0]) for fp in sample_filepath_arr]
	return sample_filepath_arr, sample_labels

def gen_label_count(labels):
	label_count = {}
	for l in labels:
		if l in label_count.keys():
			label_count[l] += 1
		else:
			label_count[l] = 1

	return label_count

def check_dir_fit(dd):
	sample_filepath_arr = get_dir_filepath_arr(dd)
	sample_labels = [int(os.path.basename(fp).split(".")[0]) for fp in sample_filepath_arr]
	
	label_count = gen_label_count(sample_labels)
	if len(label_count.keys())<2:
		return False

	return True

def fit_model_by_dir(sample_dir_or_path):
	if isinstance(sample_dir_or_path, str):
		sample_dir = Dir(path=sample_dir_path, mode="binary")
	else:
		sample_dir = sample_dir_or_path
	
	sample_filepath_arr, sample_labels = get_dir_fp_and_label_arr(sample_dir)

	if len(sample_labels)<1000:
		n_jobs = 1
	else:
		n_jobs = 2

	lssc_ml = LSSC_ML(n_jobs=n_jobs)
	lssc_ml.fit(sample_filepath_arr, lssc_params={"is_bounding":True, "n_lssc_rows":60, "n_lssc_columns":60})
	X = lssc_ml.transform()
	Y = sample_labels

	print X, Y

	label_count = gen_label_count(sample_labels)
	print "label_count:",label_count
	n_neighbors = min(label_count.items(), key=lambda x: x[1])[1]

	clf = KNC(n_neighbors=n_neighbors, weights="distance", algorithm="brute")
	clf.fit(X, Y)

	return {"clf":clf, "lssc_ml": lssc_ml}

def predict_by_model(model, predict_dir):
	# lssc = LSSC(image_string=sample_input)
	predict_filepath_arr = get_dir_filepath_arr(predict_dir)
	clf = model["clf"]
	lssc_ml = model["lssc_ml"]
	x = lssc_ml.transform(predict_filepath_arr, lssc_params={"is_bounding":True, "n_lssc_rows":60, "n_lssc_columns":60})
	print "pred_x:",x
	predict_y = clf.predict(x)[0]
	return predict_y


"""
test 
==============================================
"""


def show_image(image):
	cv2.imshow('image',image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def test_LSSC():
	# print LSSC(DIR_STD_SHAPE.path_join("0.jpg")).shape_mat
	std_lssc_0 = LSSC(DIR_STD_SHAPE.path_join("wo.jpg"), n_lssc_rows=30, n_lssc_columns=30 )
	print std_lssc_0.shape_mat

def test_LSSC_ML():
	lssc_ml = LSSC_ML(n_jobs=4)
	sample_filepath_arr = get_dir_filepath_arr(DIR_SAMPLE_SHAPE)
	sample_labels = [int(os.path.basename(fp).split(".")[0]) for fp in sample_filepath_arr]
	std_filepath_arr = get_dir_filepath_arr(DIR_STD_SHAPE)

	lssc_ml.fit(sample_filepath_arr, lssc_params={"is_bounding":True, "n_lssc_rows":60, "n_lssc_columns":60})
	exp_score(lssc_ml.transform(), sample_labels)
	# for a,b in zip(lssc_ml.transform(sample_filepath_arr),sample_labels):
	# 	print a,b



if __name__ == '__main__':
	test_LSSC()
	# test_LSSC_ML()

