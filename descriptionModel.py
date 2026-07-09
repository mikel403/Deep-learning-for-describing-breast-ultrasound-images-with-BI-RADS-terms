import torch, torchvision
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn as nn
import numpy as np
import os 


from .openImage import openImage
from .biradsModel import predict_naive


Shape=["irregular","oval","round"]
Margin=["angulated","circumscribed","spiculated","indistinct","microlobulated"]
Orientation=["parallel","no orientation","not parallel"]
Echogenicity=["anechoic","heterogeneous","hypoechoic","isoechoic"]
Posterior=["enhancement","no features","shadowing"]
Suggestivity=["complicated cyst", "simple cyst", "other"]
# Suggestivity2=["fibroadenoma", "complex cyst", "simple cyst"]
results=["benign","malignant"]

voc2=set(Shape+Margin+Orientation+Echogenicity+Posterior+Suggestivity+results)
voc2=list(voc2)
voc2.sort()


Shape.sort()
Margin.sort()
Echogenicity.sort()
Orientation.sort()
Posterior.sort()

Suggestivity.sort()

results.sort()

BIRADS=["2","3","4A","4B","4C","5"]
BIRADS.sort()

word_to_idx_shape = dict((word, idx) for idx, word in enumerate(Shape))
idx_to_word_shape = dict((idx, word) for idx, word in enumerate(Shape))

word_to_idx_margin = dict((word, idx) for idx, word in enumerate(Margin))
idx_to_word_margin = dict((idx, word) for idx, word in enumerate(Margin))

word_to_idx_orientation = dict((word, idx) for idx, word in enumerate(Orientation))
idx_to_word_orientation = dict((idx, word) for idx, word in enumerate(Orientation))

word_to_idx_echogenicity = dict((word, idx) for idx, word in enumerate(Echogenicity))
idx_to_word_echogenicity = dict((idx, word) for idx, word in enumerate(Echogenicity))

word_to_idx_posterior = dict((word, idx) for idx, word in enumerate(Posterior))
idx_to_word_posterior = dict((idx, word) for idx, word in enumerate(Posterior))

word_to_idx_suggestivity = dict((word, idx) for idx, word in enumerate(Suggestivity))
idx_to_word_suggestivity = dict((idx, word) for idx, word in enumerate(Suggestivity))

word_to_idx_results = dict((word, idx) for idx, word in enumerate(results))
idx_to_word_results = dict((idx, word) for idx, word in enumerate(results))

word_to_idx_birads = dict((word, idx) for idx, word in enumerate(BIRADS))
idx_to_word_birads = dict((idx, word) for idx, word in enumerate(BIRADS))



VGG16=torchvision.models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
for param in VGG16.parameters():
    param.requires_grad = False
    
return_layers = {'23': 'pooling'}
Encoder_VGG16 = IntermediateLayerGetter(VGG16.features, return_layers=return_layers)

class simple_encoder(nn.Module):
    def __init__(self):
        super(simple_encoder, self).__init__()
        self.conv=nn.Conv2d(1,3,3)
        self.gelu=nn.GELU()
        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.encoder=Encoder_VGG16
        
    def forward(self, inputs):
        conv_out=self.gelu(self.conv(inputs))
#         conv_out=self.conv(inputs)
        pool=nn.MaxPool2d(2)(conv_out)
        enc_out=self.encoder(pool)
        return enc_out["pooling"]

class Attention(nn.Module):
    def __init__(self, dropout=False, L2Attention=False, Gatted=False, L2dim=20,feature_dim=512,**kwargs):
        super(Attention, self).__init__()
        self.L2Attention=L2Attention
        self.Gatted=Gatted
        self.L2dim=L2dim
#         self.weight_initializer = tf.keras.initializers.glorot_normal
#         self.const_initializer = tf.keras.initializers.Zeros()
        self.dropout=dropout
        
        F2=feature_dim
        #Inicia los features que dependen de la entrada
        
        #Iniciar pesos  Atención
        if self.L2Attention or self.Gatted:
            self.w_a_1= nn.Parameter(torch.rand(F2, L2dim))
            self.b_a_1=nn.Parameter(torch.zeros(L2dim))
            
            self.w_a_2= nn.Parameter(torch.rand(L2dim,1))
            self.b_a_2=nn.Parameter(torch.zeros(1))
            
            if self.Gatted:
                self.w_a_g= nn.Parameter(torch.rand(F2, L2dim))
                self.b_a_g=nn.Parameter(torch.zeros(L2dim))
             
        else:
            w_a=nn.Parameter(torch.randn(F2, 1))
            torch.nn.init.xavier_uniform_(w_a, gain=1.0)
            b_a=nn.Parameter(torch.randn(1))
            self.params=nn.ParameterDict({"w_a": w_a,
            "b_a":b_a})
        self.dropout_layer=nn.Dropout(0.5)
        self.softmax=nn.Softmax(-1)
    def forward(self, inputs):
        input_data=inputs
        input_data=input_data.view(-1,input_data.shape[-3],input_data.shape[-2]**2)
        input_data=input_data.permute(0,-1,-2)
        #Primear atención
        #lstm
        if self.L2Attention or self.Gatted:
            Attention1=torch.tanh(reshape_matmul(input_data,self.w_a_1)+self.b_a_1) #(B,F_1,L2dim)

            if self.Gatted:
                Attention2=torch.sigmoid(reshape_matmul(input_data,self.w_a_g)+self.b_a_g) #(B,F_1,L2dim)
                Attention1=Attention1*Attention2
            Attention=reshape_matmul(Attention1,self.w_a_2)+self.b_a_2 #(B,F_1,1)
        else:
            Attention=torch.tanh(reshape_matmul(input_data,self.params["w_a"])+self.params["b_a"]) #(B,F_1,1)

#         if self.dropout and False:
#             alfa=torch.softmax(self.dropout_layer(Attention[:,:,0]))
#         else:
        alfa=self.softmax(Attention[:,:,0]) #(B,F_1)
        context = torch.sum(input_data*torch.unsqueeze(alfa, dim=2), 1) #(B,F_2)

        if self.dropout:
            context=self.dropout_layer(context)  
        return context, alfa
    
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.forma_d=nn.Linear(512,3)
        self.margen_d=nn.Linear(512,5)
        self.orientacion_d=nn.Linear(512,3)
        self.ecogenicidad_d=nn.Linear(512,4)
        self.posterior_d=nn.Linear(512,3)
    
        self.sugestividad_d=nn.Linear(512+len(voc2)-len(Suggestivity)-2,3)
        self.benignidad_d=nn.Linear(512+len(voc2)-2,2)
        torch.nn.init.xavier_uniform_(self.forma_d.weight)
        torch.nn.init.xavier_uniform_(self.margen_d.weight)
        torch.nn.init.xavier_uniform_(self.orientacion_d.weight)
        torch.nn.init.xavier_uniform_(self.ecogenicidad_d.weight)
        torch.nn.init.xavier_uniform_(self.posterior_d.weight)
        
        torch.nn.init.xavier_uniform_(self.sugestividad_d.weight)
        torch.nn.init.xavier_uniform_(self.benignidad_d.weight)
    def forward(self, inputs):
        forma=self.forma_d(inputs)
        margen=self.margen_d(inputs)
        orientacion=self.orientacion_d(inputs)
        ecogenicidad=self.ecogenicidad_d(inputs)
        posterior=self.posterior_d(inputs)
        
        caracs=torch.cat((inputs,forma,margen,orientacion,ecogenicidad,posterior),-1)
        sugestividad=self.sugestividad_d(caracs)
        caracs_f=torch.cat((caracs,sugestividad),-1)
        benignidad=self.benignidad_d(caracs_f)
        
        forma=nn.functional.softmax(forma,dim=1)
        margen=nn.functional.softmax(margen,dim=1)
        orientacion=nn.functional.softmax(orientacion,dim=1)
        ecogenicidad=nn.functional.softmax(ecogenicidad,dim=1)
        posterior=nn.functional.softmax(posterior,dim=1)
        sugestividad=nn.functional.softmax(sugestividad,dim=1)
        benignidad=nn.functional.softmax(benignidad,dim=1)

        return forma, margen, orientacion, ecogenicidad, posterior, sugestividad, benignidad
    
class Att_model(nn.Module):
    def __init__(self,dropout=False, L2Attention=False, Gatted=False, L2dim=20,feature_dim=512):
        super(Att_model, self).__init__()
        self.encoder=simple_encoder()
        self.norm=nn.BatchNorm2d(512)
        self.Att=Attention(dropout=dropout, L2Attention=L2Attention, Gatted=Gatted, L2dim=L2dim,feature_dim=feature_dim)
        self.decoder=decoder()
    def forward(self, inputs):
        enc_out=self.encoder(inputs)
        enc_out_norm=self.norm(enc_out)
        context,alfa=self.Att(enc_out_norm)
        forma, margen, orientacion, ecogenicidad, posterior, sugestividad, benignidad=self.decoder(context)
        return forma, margen, orientacion, ecogenicidad, posterior, sugestividad, benignidad
    

model_path = os.path.join('models', 'model_birads5.pth')
model=Att_model().cpu()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model=model.cpu()
model.eval()


def reshape_matmul(M3D,M2D):
    return torch.einsum('ijk,kl->ijl', M3D, M2D)

tamaño=450
def bound_img(imag):
    imag=torch.from_numpy(np.expand_dims(imag,axis=0))
    _,ancho,largo=imag.shape
    ratio=ancho/largo
    if ancho>tamaño or largo>tamaño:
        if ratio>1:
            imag=torchvision.transforms.Resize((tamaño,int(tamaño/ratio)))(imag)
        else:
            imag=torchvision.transforms.Resize((int(tamaño*ratio),tamaño))(imag)
        _,ancho,largo=imag.shape
#     imag=img_to_array(imag)/255.
#     imag=tf.image.pad_to_bounding_box(
#     imag, (tamaño-largo)//2,(tamaño-ancho)//2, tamaño, tamaño
#     )

    largo_offset=(tamaño-largo)//2
    ancho_offset=(tamaño-ancho)//2
    largo_offset_r=largo_offset if (tamaño-largo)%2==0 else largo_offset+1
    ancho_offset_d=ancho_offset if (tamaño-ancho)%2==0 else ancho_offset+1
    imag=torchvision.transforms.Pad([largo_offset,ancho_offset,largo_offset_r,ancho_offset_d])(imag)
    imag=torch.unsqueeze(imag,dim=0)
    return(imag/255.) 

def convert_to_words(desc,threshold=0.2):
    output_ejemplo=[]
    total=[]
    prob_dic = {}
    redondeada_bool=False
    hipoecoica_bool=False
    result=desc
    naive_total=[]
    for t,carac in enumerate(result):
        carac=np.squeeze(carac)
        maxi=torch.max(carac)
        result_carac=int(torch.argmax(carac))
        #No queremos guardar la benignidad
        if t!=len(result)-1:
            naive_total.append(result_carac)
        final=result_carac
        #Las únicas características que pueden no darse son la orientación y la característica posterior.
        if t==0:
            m=torch.zeros(len(Shape))
            m[final]=1
            word=idx_to_word_shape[final]
            output_ejemplo.append(word)
            if word=="oval":
                oval_bool=True
            elif word=="round":
                redondeada_bool=True
            total.append(m)
            prob_dic['shape'] = {Shape[i]: round(float(carac[i]),4) for i in range(len(Shape))}
        if t==1:
            m=torch.zeros(len(Margin))
            
            m[final]=1
            word=idx_to_word_margin[final]
            output_ejemplo.append(word)
            total.append(m)
            prob_dic['margin'] = {Margin[i]: round(float(carac[i]),4) for i in range(len(Margin))}
        if t==2:
            m=torch.zeros(len(Orientation))
            if maxi>0.3:
                m[final]=1
                if not redondeada_bool:
                    word=idx_to_word_orientation[final]
                    output_ejemplo.append(word)
                else:
                    output_ejemplo.append("no orientation")
            
            total.append(m)
            prob_dic['orientation'] = {Orientation[i]: round(float(carac[i]),4) for i in range(len(Orientation))}
        if t==3:
            m=torch.zeros(len(Echogenicity))
            
            m[final]=1
            word=idx_to_word_echogenicity[final]
            if word=="hypoechoic":
                hipoecoica_bool=True
            output_ejemplo.append(word)
            total.append(m)
            prob_dic['echogenicity'] = {Echogenicity[i]: round(float(carac[i]),4) for i in range(len(Echogenicity))}
        if t==4:
            m=torch.zeros(len(Posterior))
            
            m[final]=1
            word=idx_to_word_posterior[final]
            output_ejemplo.append(word)
            total.append(m)
            prob_dic['posterior'] = {Posterior[i]: round(float(carac[i]),4) for i in range(len(Posterior))}
        
        if t==5:
            m=np.zeros(len(Suggestivity))
            
            word=idx_to_word_suggestivity[final]
            if word=="simple cyst" and hipoecoica_bool:
                #Lo clasifica como otro
                final=1

            m[final]=1
            if word!="other":
                output_ejemplo.append(word)
            else:
                output_ejemplo.append("")
            #ELIMINO OTRO
            total.append(torch.from_numpy(np.array([m[0]]+[m[2]])))
            prob_dic['suggestivity'] = {Suggestivity[i]: round(float(carac[i]),4) for i in range(len(Suggestivity))}
        if t==6:
        
            word=idx_to_word_results[final]
            output_ejemplo.append(word)
    total=torch.cat(total)
    return output_ejemplo,naive_total, prob_dic

def results_simple(imag,model=model):
    imag=openImage(imag)
    imag=bound_img(imag)
    results=model(imag)
    words,total,prob_dic=convert_to_words(results)
    # total=total.float()
    #Los resultados a la red hay que darlos en dimensión 2
    #La salida también es dimensión 2
    birads_result=predict_naive([total])[0]
    prob_dic['birads'] = {BIRADS[i]: round(float(birads_result[i]),4) for i in range(len(BIRADS))}
    birads=np.argmax(birads_result)

    total=list(total)
    total.append(birads)
    birads_word=idx_to_word_birads[int(birads)]
    words.append(birads_word)
    return(total,words,prob_dic)