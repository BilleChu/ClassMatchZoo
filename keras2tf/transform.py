#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.models import Model
from keras.models import load_model
from keras import backend as K
import codecs
import sys
#reload(sys)
#sys.setdefaultencoding( 'utf-8' )
import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
import re

K.set_learning_phase( False )

def save_model( strSaveTo, sess, stModel, nVersion ):
	#导出tensorflow serving坯加载的模型
	if not os.path.exists( strSaveTo ):
		os.mkdir( strSaveTo )
	tf.app.flags.DEFINE_integer('model_version', nVersion, 'version number of the model.')
	tf.app.flags.DEFINE_string('work_dir', strSaveTo, 'Working directory.')

	FLAGS = tf.app.flags.FLAGS
	export_path = os.path.join( tf.compat.as_bytes(strSaveTo), tf.compat.as_bytes(str(FLAGS.model_version)))
	print( 'Exporting trained model to', export_path )
	if os.path.exists( export_path ):
		command = 'rm -r ' + export_path
		print( command )
		os.system( command )

	stSaveBuilder = saved_model_builder.SavedModelBuilder(export_path)
	inputs = {
		"X":tf.saved_model.utils.build_tensor_info(stModel.input),
	}
	# output为最终需覝的输出结果tensor
	outputs = {'prob' : tf.saved_model.utils.build_tensor_info(stModel.output)}

	signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
	legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
	stSaveBuilder.add_meta_graph_and_variables( sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={"res_censor_tf":signature,}, legacy_init_op=legacy_init_op)
	stSaveBuilder.save()

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print( 'py <stage 2 model> <save to dir> <version>' )
		sys.exit(0)

	strFromPathName = sys.argv[ 1 ]
	strToPathName = sys.argv[ 2 ]
	nVersion = int(sys.argv[ 3 ])
	print( 'from:' + strFromPathName )
	print( 'to:' + strToPathName )
	print( 'version:' + str(nVersion) )

	if not os.path.exists( strToPathName ):
		os.mkdir( strToPathName )

	stModel = load_model( strFromPathName )
	stModel.summary()
	with K.get_session() as sess:
		save_model( strToPathName, sess, stModel, nVersion )

	print( 'Done!' )


