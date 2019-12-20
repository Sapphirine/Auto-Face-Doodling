import tensorflow as tf
from keras.models import model_from_json

def init():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
        #print(loaded_model_json)
    
	loaded_model = model_from_json(loaded_model_json)
	#load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
	graph = tf.get_default_graph()


	return loaded_model,graph

def load_face():
    json_file = open('model_face.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    #print(loaded_model_json)
    
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model_face.h5")
    print("Loaded Model from disk")
    
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return loaded_model

def load_eye():
    json_file = open('model_eye.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    #print(loaded_model_json)
    
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model_eye.h5")
    print("Loaded Model from disk")
    
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return loaded_model

def load_mouth():
    json_file = open('model_mouth.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    #print(loaded_model_json)
    
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model_mouth.h5")
    print("Loaded Model from disk")
    
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return loaded_model

def load_nose():
    json_file = open('model_nose.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    #print(loaded_model_json)
    
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model_nose.h5")
    print("Loaded Model from disk")
    
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return loaded_model

def load_eye():
    json_file = open('model_ear.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    #print(loaded_model_json)
    
    loaded_model = model_from_json(loaded_model_json)
    #load weights into new model
    loaded_model.load_weights("model_ear.h5")
    print("Loaded Model from disk")
    
    #compile and evaluate loaded model
    loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return loaded_model
