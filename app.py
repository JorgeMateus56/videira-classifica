import streamlit as st
import gdown
import tensorflow as tf
@st.cache_resource
#---------------------------------------------------------------------------------------
def carrega_modelo():
  #https://drive.google.com/file/d/1VPpw18cagLfP8j6smglsCrarV0182Jkh/view?usp=drive_link
  url = 'https://drive.google.com/uc?id=1VPpw18cagLfP8j6smglsCrarV0182Jkh'
  
  gdown.download(url,'modelo_quantizado16bits.tflite')
  interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
  interpreter.allocate_tensors()
  
  return interpreter
#---------------------------------------------------------------------------------------
def main():
  st.set_page_config(
    page_title="Classifica Folhas de Videira",
    page_icon=" "
  )

st.write("# Classifica Folhas de Videira!")

  # Carrega modelo

  # Carrega imagem

  # Classifica
#--------------------------------------------------------------------------------------

if __name__==__main__:
  main()
