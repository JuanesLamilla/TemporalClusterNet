import numpy as np
import pandas as pd
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Flatten, Input

def extract(keras_model, keras_preprocess, image_list, shape, summary = False) -> pd.DataFrame:
    """
    Source:
    https://github.com/caiocarneloz/kerfex/blob/main/kerfex

    Example of usage:

    from feature_extraction import extract
    import pandas as pd
    from keras.preprocessing import image
    from keras.applications.vgg16 import VGG16 as vgg
    from keras.applications.vgg16 import preprocess_input as vgg_p

    def main():
        
        path = 'images/'
        images = ['caglaroskay.jpg', 'davidbraud.jpg', 'jessicaknowlden.jpg']
        
        image_list = []
        for img in images:
            image_list.append(image.load_img(path+img, target_size=(300, 500)))
            
        df_features = krf.extract(vgg, vgg_p, image_list, (300, 500, 3))
        
        print(df_features)
        
        
    if __name__ == "__main__":
        main()
    """
    
    model = keras_model(weights='imagenet', include_top = False)
    
    if summary:
        model.summary()

    images = []
    for img in image_list:
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = keras_preprocess(img_data)
        images.append(img_data)

    images = np.vstack(images)
    
    model_input = Input(shape=shape, name='input')
    
    output = model(model_input)
    x = Flatten(name='flatten')(output)
    
    extractor = Model(inputs=model_input, outputs=x)
    features = extractor.predict(images)

    df = pd.DataFrame.from_records(features)
    df = df.loc[:, (df != 0).any(axis=0)]
    df.columns = np.arange(0,len(df.columns))
    
    # Add image names to the dataframe
    df['image'] = [img.filename for img in image_list]

    return df