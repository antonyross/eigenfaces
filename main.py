# -*- coding: utf-8 -*-
import eigenfaces


ef = eigenfaces.EigenFaces()
ef.train("training_images")
ef.show_results()
predicted_class = ef.predict_face_in_image(0)
print(predicted_class)
eigenfaces.show_image_for(predicted_class)
