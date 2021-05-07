import streamlit as st
import pandas as pd
from stop_words import get_stop_words
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import unicodedata
import re
import nltk
import altair as alt
import pickle
import sklearn
import string
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator,get_single_color_func
import numpy as np
from io import BytesIO
import base64
nltk.download('punkt')
class SimpleGroupedColorFunc(object):
    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color for (color, words) in color_to_words.items() for word in words}
        self.default_color = default_color
    def __call__(self, word, **kwargs):
	    return self.word_to_color.get(word, self.default_color)

class Predictor():
    def __init__(self, df):
        with open('model_sent_3011','rb') as f:
            model_sent = pickle.load(f)
        self.model = model_sent
        self.df = df 
        self.text_handler = TextHandler(self.df)
        
    def make_prediction_sent(self,text):
        text = pd.Series(text)
        text = text.apply(self.text_handler.clean_text_to_sentimentation(text))
        result = self.model.predict(text)
        return result 
    #sasasa
class TextHandler():
    def __init__(self, df):
        self.df = df
        self.punctuation = '!"$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.stop_words = ['de','a','o','que','e','é','do','da','em','um','para','com','não','uma','os','no','se','na','por','mais','as','dos','como',
	'mas','ao','ele','das','à','seu','sua','ou','quando','muito','nos','já','eu','também','só','pelo','pela','até','isso','ela','entre','depois',
	'sem','mesmo','aos','seus','quem','nas','me','esse','eles','você','essa','num','nem','suas','meu','às','minha','numa','pelos','elas','qual',
	'nós','lhe','deles','essas','esses','pelas','este','dele','tu','te','vocês','vos','lhes','meus','minhas','teu','tua','teus','tuas','nosso',
	'nossa','nossos','nossas','dela','delas','esta','estes','estas','aquele','aquela','aqueles','aquelas','isto','aquilo','estou','está','estamos',
	'estão','estive','esteve','estivemos','estiveram','estava','estávamos','estavam','estivera','estivéramos','esteja','estejamos','estejam','estivesse',
	'estivéssemos','estivessem','estiver','estivermos','estiverem','hei','há','havemos','hão','houve','houvemos','houveram','houvera','houvéramos',
	'haja','hajamos','hajam','houvesse','houvéssemos','houvessem','houver','houvermos','houverem','houverei', 'houverá','houveremos','houverão',
	'houveria','houveríamos','houveriam','sou','somos','são','era', 'éramos','eram','fui','foi','fomos','foram','fora','fôramos','seja','sejamos',
	'sejam','fosse','fôssemos','fossem','for','formos','forem','serei','será','seremos','serão','seria','seríamos','seriam','tenho','tem','temos','tém',
	'tinha','tínhamos','tinham','tive','teve','tivemos','tiveram','tivera','tivéramos','tenha','tenhamos','tenham','tivesse','tivéssemos','tivessem',
	'tiver','tivermos','tiverem','terei','terá','teremos','terão','teria','teríamos','teriam','pra','ja','la','so','ai']
    
    def strip_accents(self, text):
        try:
            text = unicode(text, 'utf-8')
        except NameError:
            pass 
    
        text = unicodedata.normalize('NFD',text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
     
        return str(text)
    
    def clean_freq_word(self, text):
        text = text.lower()
        text = self.strip_accents(text)
        text = re.sub('[%s]' % re.escape(self.punctuation), '', text)
        return str(text)
    
    def clean_text_to_sentimentation(self,text):
        text = text.str.lower()
         # Letra minúscula
        print("antes do erro")
        text = self.strip_accents(text) # Removendo acentos das palavras
        print("passou strip")
        remove = string.punctuation # deixando apenas os pontos de exclamação e interrogação
        remove = remove.replace('?', '')
        remove = remove.replace('!', '')
        remove = remove.replace('#', '')
        pattern = r"[{}]".format(remove) 
        text = re.sub(pattern, "", text) 
        
        text = re.sub('\w*\d\w*', '', text) # removendo digitos
        text = re.sub('\n', '', text) # Removendo quebras de linha
        return text
    
    def to_excel(self,df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
        df.to_excel(writer, sheet_name='Planilha1',index = False)
        writer.save()
        processed_data = output.getvalue()
        return processed_data
 
    def get_table_download_link(self,df):
        val = self.to_excel(df)
        b64 = base64.b64encode(val)
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download</a>'
            
    def word_tokenizer(self, text,top_n):
        text = text.str.cat(sep = ' ')
        words_text = nltk.tokenize.word_tokenize(text)
        words_dist = nltk.FreqDist(words_text)
        words_dist_stop = nltk.FreqDist(w for w in words_text if w not in self.stop_words)
        words_common = pd.DataFrame(words_dist_stop.most_common(top_n), columns=['Palavra', 'Quantidade'])
        return words_common
    
    def get_freq_words(self, reviews_done, sent, top_n=10):
        df_sent = reviews_done[reviews_done['Sentimentação'] == sent]
        txt = df_sent['Review']
        txt = txt.apply(self.clean_freq_word)    
        txt = txt.str.cat(sep = ' ')
        words_txt = nltk.tokenize.word_tokenize(txt)
        words_dist = nltk.FreqDist(words_txt)
        words_dist_stop = nltk.FreqDist(w for w in words_txt if w not in self.stop_words)
        words_common = pd.DataFrame(words_dist_stop.most_common(top_n), columns = ['Palavra','Quantidade'])
		
        words_common['Sentimentação'] = sent
        return words_common

class Application():
    def __init__(self):
        self.df = pd.DataFrame({})
        self.text_handler = TextHandler(self.df)
        self.predictor = Predictor(self.df)
    def render_header(self):
        st.image('assets/di_produto_rank.png')
        st.markdown('<h1 style = "border-radius: 50px; color: #FFFFFF;text-align:center;text-transform: uppercase; background:-webkit-linear-gradient(#FD6E1B, #DB2F40);">Análise de Comentários - MVP</h1>',unsafe_allow_html=True)
        st.text("")
        st.text("")
        st.markdown("<p style = 'text-align: center; color: #7E7E7E; font-size: 20px'>Plataforma em estágio MVP, que permite uma análise exploratória e aplicação de modelos de sentimentação em um volume de comentários, obtendo insight’s sobre o engajamento da publicação. </p>", unsafe_allow_html=True )


    def render_load_file(self):
        flag = False
        uploaded_file = st.file_uploader(label = "Arraste o arquivo", type = 'xlsx')
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file, usecols = ['Usuario','Comentario','Repeticao'], sheet_name = 'Comentários')
            self.df = df
            flag = True
        return flag
    
    def render_frequency_words(self):
        st.markdown("<h2 style ='text-align: center; color: #FD6E1B;'> Palavras mais Frequentes</h2>",unsafe_allow_html=True)
        st.markdown("<p style = 'text-align: center; color: #7E7E7E; font-size:20px'> Levantamento da frequência de repetição de palavras. Selecione no slider abaixo, a quantidade de palavras para exibição. Clique no botão abaixo, para download da planilha com esta métrica.</p>", unsafe_allow_html=True )
        col1, col2 = st.beta_columns(2)
        
        with col1:
            st.markdown("<h3 style = 'text-align: center'> Total de Comentários </h3>", unsafe_allow_html = True)
            comentarios_totais = self.df['Comentario'].shape[0]
            st.markdown("<p style = 'text-align: center'>" + str(comentarios_totais) + '</p>', unsafe_allow_html=True)
        with col2:
            st.markdown("<h3 style = 'text-align: center'> Comentários Únicos </h3>", unsafe_allow_html=True)
            comentarios_unicos = self.df['Comentario'].unique().shape[0]
            st.markdown('<p style = "text-align: center">' + str(comentarios_unicos) + '</p>', unsafe_allow_html=True)
            
        col1, col2  = st.beta_columns(2)
        st.markdown('<p style ="color: #7E7E7E; text-align: center; font-size: 15px"> Arraste para o valor desejado, entre 0 à 30: </p>',unsafe_allow_html = True),
        top_n = st.slider("",0, 30,10)
        text = self.df['Comentario']
        text = text.apply(self.text_handler.clean_freq_word)
        words_common = self.text_handler.word_tokenizer(text,top_n)
        palette = ["#FD6E1B"]
        selector = alt.selection_single(empty = 'all', fields = ['Palavra'])
        base = alt.Chart(words_common).add_selection(selector)
        c= base.mark_bar().\
                    encode(x=alt.Y('Palavra',sort='-y'),
                           y='Quantidade',
                           tooltip = ['Palavra','Quantidade'],
                           color = alt.condition(selector, 'Palavra:N',
                                                 alt.value('lightgray'), legend = None,
                                                 scale = alt.Scale(range = palette))).\
                    configure_axisX(labelAngle=315).properties(title = 'Palavras Mais Frequentes')
        st.altair_chart(c, use_container_width=True)
        st.markdown(self.text_handler.get_table_download_link(words_common), unsafe_allow_html=True)

    def render_hashtags_users_counters(self):
        st.markdown("<h3 style = 'text-align: center; color: #FD6E1B; font-size: 30px'> Contador de Hashtags </h3>",unsafe_allow_html= True)
        st.markdown("<p style = 'text-align: center; color: #7E7E7E; font-size:20px'> Levantamento da frequência de hashtags e usuário.Selecione no slider abaixo, a quantidade de palavras para exibição. Clique no botão abaixo, para download da planilha com esta métrica.</p>", unsafe_allow_html=True )

        st.markdown('<p style ="color: #7E7E7E; text-align: center; font-size: 15px"> Arraste para o valor desejado, entre 0 à 30: </p>',unsafe_allow_html = True),
        top_numbers = st.slider('', 0, 30,10, key ='2')
        txt = self.df['Comentario']
            
        txt_hash = txt.str.cat(sep = " ")
            
        hashtags = [i for i in txt_hash.split() if i.startswith('#')]
            
        hashtags_dist = nltk.FreqDist(hashtags)
            
        hashtags_common = pd.DataFrame(hashtags_dist.most_common(top_numbers), columns=['Palavra', 'Quantidade'])
        col1, col2 = st.beta_columns([3,1])
        with col1:    
            c = alt.Chart(hashtags_common).mark_bar().encode(
                    x = alt.Y("Palavra", sort = '-y',axis = alt.Axis(title = 'Hashtag',grid = False)),
                    y = 'Quantidade', tooltip = ['Palavra','Quantidade'],color = alt.value("#FD6E1B"),
            ).properties(title = 'Hashtags mais mencionadas', width = 400,
                            height = 400).configure_axisX(labelAngle=315)
            st.altair_chart(c,use_container_width = True)
        with col2:
            st.table(hashtags_common)
            st.markdown(self.text_handler.get_table_download_link(hashtags_common), unsafe_allow_html=True)
        #########################
        st.markdown("<h3 style = 'text-align: center; color: #FD6E1B; font-size:30px'> Usuários mais mencionados</h3>", unsafe_allow_html=True )

        #p = re.compile(r'@([^\s:]+)')
        col1, col2 = st.beta_columns([3,1])
        txt_mentions = txt.str.cat(sep = " ")
        mentions = [i for i in txt_mentions.split() if i.startswith('@')]
        mentions_list = nltk.FreqDist(mentions)
        mentions_list = pd.DataFrame(mentions_list.most_common(top_numbers), columns=['Palavra', 'Quantidade'])
        with col1:
            c = alt.Chart(mentions_list).mark_bar().encode(
                    x = alt.Y("Palavra", sort = '-y',axis = alt.Axis(title = 'Usuário',grid = False)),
                    y = 'Quantidade', tooltip = ['Palavra','Quantidade'], color = alt.value("#FD6E1B")
            ).configure_axisX(labelAngle=315).properties(width = 400,
                            height = 400).properties(title = 'Usuários mais mencionados')
                
            st.altair_chart(c,use_container_width = True)
        with col2:
            st.markdown("<p style = 'text-align: center'>Preview Usuários mais mencionados</p>", unsafe_allow_html= True)
            st.table(mentions_list.head(5))
            st.markdown(self.text_handler.get_table_download_link(mentions_list), unsafe_allow_html=True)

    def render_sentimentation(self):
        def make_prediction_sent(text):   
            with open('model_sent_3011','rb') as f:
                model_sent = pickle.load(f)
            text = pd.Series(text)
            text = text.apply(clean_text_round_1)
            result = model_sent.predict(text)
            return result
        # Função limpeza de texto
        def clean_text_round_1(text):
            text = text.lower() # Letra minúscula
            text = self.text_handler.strip_accents(text) # Removendo acentos das palavras
            
            remove = string.punctuation # deixando apenas os pontos de exclamação e interrogação
            remove = remove.replace('?', '')
            remove = remove.replace('!', '')
            remove = remove.replace('#', '')
            pattern = r"[{}]".format(remove) 
            text = re.sub(pattern, "", text) 
            
            text = re.sub('\w*\d\w*', '', text) # removendo digitos
            text = re.sub('\n', '', text) # Removendo quebras de linha
            return text
        st.markdown("<h2 style = 'text-align: center; color: #FD6E1B'> Sentimentação </h3>", unsafe_allow_html=True)
        st.markdown("<p style = 'text-align: center; color: #7E7E7E; font-size:20px'> Sentimentação dos comentários dos usuários além da nuvem de palavras com os termos mais mencionados. </p>", unsafe_allow_html=True )

        reviews = self.df['Comentario']
        reviews_sent = make_prediction_sent(reviews)
        reviews_done = pd.DataFrame({'Review': self.df['Comentario'],
		'Sentimentação': reviews_sent})
          
        st.markdown('<style>.positivos{color:#218d21; font-size: 24px; font-weight: bold, border-right: 5px solid orange;}</style>', unsafe_allow_html=True)
        st.markdown('<style>.negatives{color: red; font-size: 24px; font-weight: bold}</style>', unsafe_allow_html=True)
        st.markdown('<style>.neutral{color: #0BD5D2; font-size: 24px; font-weight: bold}</style>', unsafe_allow_html=True)
        st.markdown('<style>.mixed{color: #FE7B2F; font-size: 24px; font-weight: bold}</style>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.beta_columns(4)
        with col1:
            st.subheader("Positivos")
            positivos = reviews_done[reviews_done['Sentimentação'] == 'Positive'].count()
            st.markdown('<p class="positivos">' + str(positivos[0]) + '</p>', unsafe_allow_html=True)
        with col2:
            st.subheader("Negativos")
            negativos = reviews_done[reviews_done['Sentimentação'] == 'Negative'].count()
            st.markdown('<p class="negatives">' + str(negativos[0]) + '</p>', unsafe_allow_html=True)
        with col3:
            st.subheader('Neutros')
            neutral = reviews_done[reviews_done['Sentimentação'] == 'Neutral'].count()
            st.markdown('<p class="neutral">' + str(neutral[0]) + '</p>', unsafe_allow_html=True)
        with col4:
            st.subheader("Mixed")
            mixed = reviews_done[reviews_done['Sentimentação'] == 'Mixed'].count()
            st.markdown('<p class="mixed">' + str(mixed[0]) + '</p>', unsafe_allow_html=True)
        return reviews_done
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    
    def render_wordcloud(self, reviews_done):
        freq_pos = self.text_handler.get_freq_words(reviews_done, 'Positive',reviews_done.shape[0])
        freq_neg = self.text_handler.get_freq_words(reviews_done,'Negative',reviews_done.shape[0])
        freq_mix = self.text_handler.get_freq_words(reviews_done,'Mixed',reviews_done.shape[0])
        freq_neu = self.text_handler.get_freq_words(reviews_done,'Neutral',reviews_done.shape[0])
        
        all_sent = [freq_pos,freq_neu,freq_mix,freq_neu]
        freq_all = pd.concat(all_sent).reset_index(drop = True)
        freq_unique = freq_all.groupby('Palavra')['Quantidade','Sentimentação'].max().reset_index()
        
        mask_pos = freq_unique['Sentimentação'] == 'Positive'
        list_pos = freq_unique[mask_pos]['Palavra']
        
        mask_neg = freq_unique['Sentimentação'] == 'Negative'
        list_neg = freq_unique[mask_neg]['Palavra']
        
        mask_mix = freq_unique['Sentimentação'] == 'Mixed'
        list_mix = freq_unique[mask_mix]['Palavra']
        
        mask_neu = freq_unique['Sentimentação'] == 'Neutral'
        list_neu = freq_unique[mask_neu]['Palavra']
        
        color_to_words = {
            "#218d21": list_pos,
            "red": list_neg,
            "#FE7B2F": list_mix,
            "#0BD5D2": list_neu
        }
        
        default_color = 'grey'
        
        grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)
        records = freq_all[['Palavra','Quantidade']].to_records(index = False)
        frequencies = dict(list(records))
        wc = WordCloud(collocations=False, background_color= 'white').fit_words(frequencies)
        wc.recolor(color_func=grouped_color_func)
        plt.figure(figsize = (12,10))
        plt.imshow(wc,interpolation='bilinear')
        plt.axis('off')
        st.image(wc.to_array(),width = 600,)
        
    def render_app(self):
        st.set_page_config(page_title='Análise de Comentários - MVP',
                           layout='centered', page_icon= "assets/di_produto_rank2.png",
                           initial_sidebar_state='auto')
        
        self.render_header()
        flag = self.render_load_file()
        if flag != False:
            st.image("assets/horizontal_line.png")
            self.render_frequency_words()
            st.image("assets/horizontal_line.png")
            self.render_hashtags_users_counters()
            st.image("assets/horizontal_line.png")
            df = self.render_sentimentation()
            self.render_wordcloud(df)
        
        
app = Application()
app.render_app()
