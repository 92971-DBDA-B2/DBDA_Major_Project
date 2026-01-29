# DBDA_Major_Project(Ojas Varshney & Mohammad Amaan Ali)

A CNN-based system that captures faces and performs real-time recognition to automatically mark attendance.

Flow of Running the programs

python capture_faces.py -->

python preprocess_faces.py -->

python augmented_faces.py -->

python prepare_dataset.py -->

python train_model.py -->

streamlit run streamlit_app.py

keep in mind create the directory as per the directory names given in the codes.
Also keep in mind to use high quality Imgaes and high quality Camera to capture Images and also while doing facial Recognition

We have also used Data Augmentation as well as a pretrained model MobileNetV2 which comes along automatically while installing keras for python libary.
You can also change the model used from pre-trained CNN model to any Model of Your choice just keep in mind tp change the model name as well as the directrory of its storage.
