#import dlib
import face_detect as fd
import In_Program as ip
import onnx, threading
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import onnxruntime as ort

from onnx_tf.backend import prepare
from imutils import face_utils

def Main(queue, msg_queue):
    #region init
    # load the model, create runtime session & get input variable name
    onnx_path = 'model/ultra_light_640.onnx'
    onnx_model = onnx.load(onnx_path)
    predictor = prepare(onnx_model)
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    import dlib
    shape_predictor = dlib.shape_predictor('model/shape_predictor_68_face__landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))
    #endregion

    try:
        with tf.Graph().as_default():
            with tf.Session() as sess:
                ip.In_Program_M(sess, ort_session, input_name, fa, queue, msg_queue)

    except Exception as e:
        print(e)