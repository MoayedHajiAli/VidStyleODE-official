import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class MFMOD(nn.Module):
    '''
    Multi-Feature Modulation Block
    '''
    def __init__(self, modulation_shape):
        super().__init__()
        self.blending_gamma = nn.Parameter(torch.zeros(*modulation_shape), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(*modulation_shape), requires_grad=True)

    def forward(self, w1_gamma, w1_beta, w2_gamma, w2_beta):
        # B x n_layer x 512
        gamma_alpha = torch.sigmoid(self.blending_gamma)
        beta_alpha = torch.sigmoid(self.blending_beta)

        factor = gamma_alpha * w1_gamma + (1 - gamma_alpha) * w2_gamma
        bias = beta_alpha * w1_beta + (1 - beta_alpha) * w2_beta
        return factor, bias

class TripleMFModulationModule(nn.Module):
    def __init__(self, layernum, attr_vec_dim1, attr_vec_dim2, attr_vec_dim3, mod_shape):
        super(TripleMFModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = nn.Linear(512, 512)
        self.norm = nn.LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.attr_vec_dim1 = attr_vec_dim1
        self.attr_vec_dim2 = attr_vec_dim2
        self.attr_vec_dim3 = attr_vec_dim3

        self.gamma_function_1 = nn.Sequential(nn.Linear(attr_vec_dim1, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.beta_function_1 = nn.Sequential(nn.Linear(attr_vec_dim1, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        
        self.gamma_function_2 = nn.Sequential(nn.Linear(attr_vec_dim2, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.beta_function_2 = nn.Sequential(nn.Linear(attr_vec_dim2, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        
        self.gamma_function_3 = nn.Sequential(nn.Linear(attr_vec_dim3, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.beta_function_3 = nn.Sequential(nn.Linear(attr_vec_dim3, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        
        self.combine_modulation_12 = MFMOD(mod_shape)
        self.combine_modulation_r3 = MFMOD(mod_shape)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, embd1, embd2, embd3, cut_flag):
        # Embedding: B x L x 512
        x = self.fc(x)
        x = self.norm(x) 	
        if cut_flag:
            return x
        # print(embedding.shape)
        gamma_1 = self.gamma_function_1(embd1.float())
        beta_1 = self.beta_function_1(embd1.float())

        gamma_2 = self.gamma_function_2(embd2.float())
        beta_2 = self.beta_function_2(embd2.float())

        gamma_3 = self.gamma_function_3(embd3.float())
        beta_3 = self.beta_function_3(embd3.float())

        r_gamma, r_beta = self.combine_modulation_12(gamma_1, beta_1, gamma_2, beta_2)
        gamma, beta = self.combine_modulation_r3(r_gamma, r_beta, gamma_3, beta_3)

        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out

class MFModulationModule(nn.Module):
    def __init__(self, layernum, attr_vec_dim1, attr_vec_dim2, mod_shape):
        super(MFModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = nn.Linear(512, 512)
        self.norm = nn.LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.attr_vec_dim1 = attr_vec_dim1
        self.attr_vec_dim2 = attr_vec_dim2

        self.gamma_function_1 = nn.Sequential(nn.Linear(attr_vec_dim1, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.beta_function_1 = nn.Sequential(nn.Linear(attr_vec_dim1, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        
        self.gamma_function_2 = nn.Sequential(nn.Linear(attr_vec_dim2, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.beta_function_2 = nn.Sequential(nn.Linear(attr_vec_dim2, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        
        self.combine_modulation = MFMOD(mod_shape)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, embd1, embd2, cut_flag):
        # Embedding: B x L x 512

        x = self.fc(x)
        x = self.norm(x) 	
        if cut_flag:
            return x
        
        gamma_1 = self.gamma_function_1(embd1.float())
        beta_1 = self.beta_function_1(embd1.float())

        gamma_2 = self.gamma_function_2(embd2.float())
        beta_2 = self.beta_function_2(embd2.float())
        
        gamma, beta = self.combine_modulation(gamma_1, beta_1, gamma_2, beta_2)

        # TODO: experiment with adding 
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out

# Taken from HairCLIP 
# https://github.com/wty-ustc/HairCLIP/blob/main/mapper/latent_mappers.py
class ModulationModule(nn.Module):
    def __init__(self, layernum, attr_vec_dim):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = nn.Linear(512, 512)
        self.norm = nn.LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.attr_vec_dim = attr_vec_dim
        self.gamma_function = nn.Sequential(nn.Linear(attr_vec_dim, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.beta_function = nn.Sequential(nn.Linear(attr_vec_dim, 512), nn.LayerNorm([512]), nn.LeakyReLU(), nn.Linear(512, 512))
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, embedding, cut_flag):
        # Embedding: B x L x 512

        x = self.fc(x)
        x = self.norm(x) 	
        if cut_flag:
            return x
        
        gamma = self.gamma_function(embedding.float())
        beta = self.beta_function(embedding.float())

        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out


class TripleAttributeMapperModule(nn.Module):
    def __init__(self, attr_vec_dim1, attr_vec_dim2, attr_vec_dim3, layernum, mod_shape, num_of_layers=5, modtype='mfmod'):
        super(TripleAttributeMapperModule, self).__init__()
        self.modtype = modtype
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        if modtype == 'mfmod':
            self.modulation_module_list = nn.ModuleList([TripleMFModulationModule(self.layernum, attr_vec_dim1, attr_vec_dim2, attr_vec_dim3, mod_shape) for i in range(num_of_layers)])
        elif modtype == 'cat':    
            self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum, attr_vec_dim1+attr_vec_dim2+attr_vec_dim3) for i in range(num_of_layers)])
        else:
            raise "unidentified mod type"

    def forward(self, x, embedding1, embedding2, embedding3, cut_flag=False):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            if self.modtype == 'mfmod':
                x = modulation_module(x, embedding1, embedding2, embedding3, cut_flag)        
            elif self.modtype == 'cat':
                x = modulation_module(x, torch.cat((embedding1, embedding2, embedding3), -1), cut_flag)        
        return x

class DoubleAttributeMapperModule(nn.Module):
    def __init__(self, attr_vec_dim1, attr_vec_dim2, layernum, mod_shape, num_of_layers, modtype='mfmod'):
        super(DoubleAttributeMapperModule, self).__init__()
        self.modtype = modtype
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        if modtype == 'mfmod':
            self.modulation_module_list = nn.ModuleList([MFModulationModule(self.layernum, attr_vec_dim1, attr_vec_dim2, mod_shape) for i in range(num_of_layers)])
        elif modtype == 'cat':    
            self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum, attr_vec_dim1+attr_vec_dim2) for i in range(num_of_layers)])
        else:
            raise "unidentified mod type"

    def forward(self, x, embedding1, embedding2, cut_flag=False):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            if self.modtype == 'mfmod':
                x = modulation_module(x, embedding1, embedding2, cut_flag)        
            elif self.modtype == 'cat':
                x = modulation_module(x, torch.cat((embedding1, embedding2), -1), cut_flag)        
        return x

class AttributeMapperModule(nn.Module):
    def __init__(self, attr_vec_dim, layernum, num_of_layers):
        super(AttributeMapperModule, self).__init__()
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum, attr_vec_dim) for i in range(num_of_layers)])

    def forward(self, x, embedding, cut_flag=False):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            x = modulation_module(x, embedding, cut_flag)        
        return x



class AttributeMapperSubModule(nn.Module):
    def __init__(self,
                attr_vec_dim,
                n_layer,
                use_mapper,
                cut_flag,
                mod_shape,
                modtype='mfmod',
                num_of_layers=5,
                predict_delta=True):
        super(AttributeMapperSubModule, self).__init__()
        self.use_mapper = np.array(use_mapper)
        self.cut_flag = cut_flag

        self.attr_vec_dim = attr_vec_dim
        self.predict_delta = predict_delta
        self.mapper = None
        
        if sum(self.use_mapper * 1) == 3:
            self.mapper = TripleAttributeMapperModule(self.attr_vec_dim[0], self.attr_vec_dim[1], self.attr_vec_dim[2], n_layer, mod_shape=mod_shape, modtype=modtype, num_of_layers=num_of_layers)
        elif sum(self.use_mapper * 1) == 2:
            self.mapper = DoubleAttributeMapperModule(self.attr_vec_dim[self.use_mapper][0], self.attr_vec_dim[self.use_mapper][1], n_layer, mod_shape=mod_shape, modtype=modtype, num_of_layers=num_of_layers)
        elif sum(self.use_mapper * 1) == 1:
            self.mapper = AttributeMapperModule(self.attr_vec_dim[self.use_mapper][0], n_layer, num_of_layers=num_of_layers)
            
    
    def forward(self, x, attribute_vectors):
        attribute_vectors = [att for flag, att in zip(self.use_mapper, attribute_vectors) if flag]
        for i in range(len(attribute_vectors)):
            if attribute_vectors[i].shape[1] != 1:
                attribute_vectors[i] = attribute_vectors[i].unsqueeze(1).repeat(1, x.shape[1], 1)
        
        
        # coarse mapping
        if sum(self.use_mapper * 1) >= 1:
            x = self.mapper(x, *attribute_vectors, cut_flag=self.cut_flag)
        else:
            if self.predict_delta:
                x = torch.zeros_like(x).detach()
                x.requires_grad = False
        
        return x

class TripleAttributeMapper(nn.Module):
    def __init__(self,
                use_coarse_mapper,
                use_medium_mapper,
                use_fine_mapper,
                coarse_cut_flag,
                medium_cut_flag,
                fine_cut_flag,
                attr_vec_dims, 
                mod_shape = [[1], [1], [1]], 
                num_of_layers = [5, 5, 5],
                modtype='mfmod',
                predict_delta=True):
        super(TripleAttributeMapper, self).__init__()
        self.use_coarse_mapper = np.array(use_coarse_mapper)
        self.use_medium_mapper = np.array(use_medium_mapper)
        self.use_fine_mapper = np.array(use_fine_mapper)
        self.coarse_cut_flag = coarse_cut_flag
        self.medium_cut_flag = medium_cut_flag
        self.fine_cut_flag = fine_cut_flag

        self.attr_vec_dims = np.array(attr_vec_dims)
        self.predict_delta = predict_delta
        
        self.coarse_mapping = AttributeMapperSubModule(self.attr_vec_dims, 4, self.use_coarse_mapper, self.coarse_cut_flag, mod_shape=mod_shape[0], modtype=modtype, num_of_layers=num_of_layers[0])
        self.medium_mapping = AttributeMapperSubModule(self.attr_vec_dims, 6, self.use_medium_mapper, self.medium_cut_flag, mod_shape=mod_shape[1], modtype=modtype, num_of_layers=num_of_layers[1])
        self.fine_mapping = AttributeMapperSubModule(self.attr_vec_dims, 8, self.use_fine_mapper, self.fine_cut_flag, mod_shape=mod_shape[2], modtype=modtype, num_of_layers=num_of_layers[2])
    
    def forward(self, x, attribute_vector1, attribute_vector2=None, attribute_vector3=None):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:10, :]
        x_fine = x[:, 10:, :]
        
        
        x_coarse = self.coarse_mapping(x_coarse, [attribute_vector1, attribute_vector2, attribute_vector3])
        x_medium = self.medium_mapping(x_medium, [attribute_vector1, attribute_vector2, attribute_vector3])
        x_fine = self.fine_mapping(x_fine, [attribute_vector1, attribute_vector2, attribute_vector3])
            
        output = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        return output

class AttentionAttributeMapper(nn.Module):
    def __init__(self,
                use_coarse_mapper,
                use_medium_mapper,
                use_fine_mapper,
                coarse_cut_flag,
                medium_cut_flag,
                fine_cut_flag,
                attr_vec_dims, 
                mod_shape = [[1], [1], [1]], 
                num_of_layers = [5, 5, 5],
                modtype='mfmod',
                predict_delta=True,
                inversion_layer_num = [4, 6, 8]):
        super(AttentionAttributeMapper, self).__init__()
        self.use_coarse_mapper = np.array(use_coarse_mapper)
        self.use_medium_mapper = np.array(use_medium_mapper)
        self.use_fine_mapper = np.array(use_fine_mapper)
        self.coarse_cut_flag = coarse_cut_flag
        self.medium_cut_flag = medium_cut_flag
        self.fine_cut_flag = fine_cut_flag

        self.attr_vec_dims = np.array(attr_vec_dims)
        self.predict_delta = predict_delta
        
        self.inversion_layer_num = inversion_layer_num
        self.coarse_mapping = AttributeMapperSubModule(self.attr_vec_dims, inversion_layer_num[0], self.use_coarse_mapper, self.coarse_cut_flag, mod_shape=mod_shape[0], modtype=modtype, num_of_layers=num_of_layers[0])
        self.medium_mapping = AttributeMapperSubModule(self.attr_vec_dims, inversion_layer_num[1], self.use_medium_mapper, self.medium_cut_flag, mod_shape=mod_shape[1], modtype=modtype, num_of_layers=num_of_layers[1])
        self.fine_mapping = AttributeMapperSubModule(self.attr_vec_dims, inversion_layer_num[2], self.use_fine_mapper, self.fine_cut_flag, mod_shape=mod_shape[2], modtype=modtype, num_of_layers=num_of_layers[2])
    
    def forward(self, x, conditional_vectors):
        x_coarse = x[:, :self.inversion_layer_num[0], :]
        x_medium = x[:, self.inversion_layer_num[0]:self.inversion_layer_num[0]+self.inversion_layer_num[1], :]
        x_fine = x[:, -self.inversion_layer_num[-1]:, :]
        
        coarse_att_vec, mid_att_vec, fine_att_vec = conditional_vectors
        
        x_coarse = self.coarse_mapping(x_coarse, [coarse_att_vec, None, None])
        x_medium = self.medium_mapping(x_medium, [mid_att_vec, None, None])
        x_fine = self.fine_mapping(x_fine, [fine_att_vec, None, None])
            
        output = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        return output