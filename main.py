# -*- coding: utf-8 -*-
import eigenfaces


ef = eigenfaces.EigenFaces()
ef.train("training_images")
ef.show_results()
print(ef.predict_face_in_image(0))
