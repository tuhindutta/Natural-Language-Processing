import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button

# Imports for ML model works
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Pickle import and bodel import
import pickle
infile = open('./Fake_News/fake_news_model','rb')
model = pickle.load(infile)
infile.close()


class FakeNewsGrid(GridLayout):
    def __init__(self,**kwargs):
        super(FakeNewsGrid, self).__init__()
        self.cols = 3

        self.add_widget(Label(text="News:"))
        self.n_title = TextInput()
        self.add_widget(self.n_title)

        self.press = Button(text='Classify')
        self.press.bind(on_press=self.classify)
        self.add_widget(self.press)



    def classify(self,instance):
        target = self.n_title.text        
   
        new = re.sub('[^a-zA-Z]', ' ',target)        
        new = new.lower()
        new = new.split()        
        new = [ps.stem(i) for i in new if i not in stopwords.words('english')]        
        new = ' '.join(new)
        pred = model.predict([new])    

        if pred==1:
            print('Its fake')
        else:
            print('Its true')

        
     
class FakeNews(App):
    def build(self):
        return FakeNewsGrid()


if __name__ == '__main__':
    FakeNews().run()