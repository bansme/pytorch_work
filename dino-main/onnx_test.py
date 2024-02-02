# -*- coding: utf-8 -*-
import onnx
import onnxruntime as rt
import numpy as  np

# Load the ONNX model
model = onnx.load("dino-main/weights/dino_resnet50.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))


data = np.array(np.random.randn(1,3,224,224))
sess = rt.InferenceSession("dino-main/weights/dino_resnet50.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name:data.astype(np.float32)})[0]
print(pred_onx)
print(np.argmax(pred_onx))