# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 14:00:20 2022

@author: benda
"""

import config
import numpy as np
import os
import re


def key_sort_files(value):
    #from: https://stackoverflow.com/a/59175736/15147410
    """Extract numbers from string and return a tuple of the numeric values"""
    return tuple(map(int, re.findall('\d+', value)))

def get_latex_figures(dataset = "mad_ous", fig_type = 'mask_overlay', width = "0.19", scale = "0.45", info_txt='Predicted masks'):

    if dataset == 'mad_ous':
         dataset_name = "MAD OUS"
    elif dataset == 'mesa':
        dataset_name = "MESA"
        
    if fig_type == 'masks':
        folder = f'C:/Users/benda/Documents/Jobb_Simula/MAD_motion/{dataset.upper()}/{dataset.upper()}_mask_original_2D'
    elif fig_type == 'mask_overlay':
        folder = f'C:/Users/benda/Documents/Jobb_Simula/MAD_motion/{dataset.upper()}/{dataset.upper()}_mask_overlay'
    elif fig_type == 'overlays':
        folder = f"C:/Users/benda/Documents/Jobb_Simula/MAD_motion/{dataset.upper()}/{dataset.upper()}_comp_lvrv_2D"
    
    subjects = sorted(os.listdir(folder), key=key_sort_files)
    subject_folders = [os.path.join(folder, subject) for subject in subjects]
    
    out = ""
    for s,subject in enumerate(subjects):
    # for s,subject in enumerate([subjects[0]]):
        
        files = sorted(os.listdir(subject_folders[s]), key=key_sort_files)
        # print(files)
        
        out += """\\begin{figure}[H]
    \centering\n"""
        
        for i,f in enumerate(files):
            if '.png' in f and '_gt' not in f:
                # print(f)
                num = re.findall('^[0-9]+', f)[0]
                ed_or_es = ('' if fig_type in ['masks', 'mask_overlay'] else (', ED' if i%4==0 else ', ES'))
                
                out += "	\\begin{subfigure}[b]{"+width+"\\textwidth}\n"
                out += "		\\centering\n"
                out += "		\\includegraphics[scale="+scale+"]{figures/"+dataset.upper()+"/"+fig_type.replace('_overlay', '')+"/" + f"{subject}/{f}" + "}\n"
                out += "		\\caption{" + f"{ re.findall('^[0-9]+', f)[0] }{ed_or_es}" + "}\n"
                out += "	\\end{subfigure}\n"
                # re.Match(f, "^[0-9]+")
                # print(out)
                # out = out.format([subject, f, f[:3]])
        out += "	\caption{"+info_txt+" for subject " + subject + " in the " + dataset_name + " dataset.}\n"
        out += "	\\label{fig:"+fig_type+"_" + subject + "}\n"
        out += "\\end{figure}\n\n"
        
    return out



def get_latex_subfigures0(dataset = "mad_ous", fig_type = '_comp_lvrv_2D', width = "0.19", scale = "0.45", info_txt='Predicted masks'):

    # dataset = "mad_ous"
    if dataset == 'mad_ous':
         dataset_name = "MAD OUS"
    elif dataset == 'mesa':
        dataset_name = "MESA"
        
    if fig_type == 'masks':
        folder = f'C:/Users/benda/Documents/Jobb_Simula/MAD_motion/{dataset.upper()}/{dataset.upper()}_mask_original_2D'
    elif fig_type == '_mask_overlay':
        folder = f'C:/Users/benda/Documents/Jobb_Simula/MAD_motion/{dataset.upper()}/{dataset.upper()}_mask_overlay'
    elif fig_type == '_comp_lvrv_2D':
        folder = f"C:/Users/benda/Documents/Jobb_Simula/MAD_motion/{dataset.upper()}/{dataset.upper()}_comp_lvrv_2D"
    
    subjects = sorted(os.listdir(folder), key=key_sort_files)
    # print(subjects)
    
    subject_folders = [os.path.join(folder, subject) for subject in subjects]
    # print(subject_folders)
    
    out = ""
    for s,subject in enumerate(subjects):
    # for s,subject in enumerate([subjects[0]]):
        
        files = sorted(os.listdir(subject_folders[s]), key=key_sort_files)
        # print(files)
        
        out += """\\begin{figure}[H]
    \centering\n"""
        
        for i,f in enumerate(files):
            if '.png' in f and '_gt' not in f:
                # print(f)
                num = re.findall('^[0-9]+', f)[0]
                ed_or_es = ('' if fig_type in ['masks', 'mask_overlay'] else (', ED' if i%4==0 else ', ES'))
                
                out += "	\\begin{subfigure}[b]{"+width+"\\textwidth}\n"
                out += "		\\centering\n"
                out += "		\\includegraphics[scale="+scale+"]{"+dataset.upper()+"/"+dataset.upper()+fig_type+"/" + f"{subject}/{f}" + "}\n"
                out += "		\\caption{" + f"{ re.findall('^[0-9]+', f)[0] }{ed_or_es}" + "}\n"
                out += "	\\end{subfigure}\n"
                # re.Match(f, "^[0-9]+")
                # print(out)
                # out = out.format([subject, f, f[:3]])
        # out += "	\caption{"+info_txt+" for subject " + subject + " in the " + dataset_name + " dataset.}\n"
        out += "	\\label{fig:"+fig_type+"_" + subject + "}\n"
        out += "\\end{figure}\n\\newpage\n\n"
        
    return out


def get_latex_figures0(dataset = "mad_ous", fig_type = 'overlays', info_txt='Predicted masks', extra_caption=''):

    if dataset == 'mad_ous':
         dataset_name = "MAD OUS"
         es = 0
    elif dataset == 'mesa':
        dataset_name = "MESA"
        es = 10

    folder = f'C:/Users/benda/Documents/Jobb_Simula/MAD_motion/{dataset.upper()}/{dataset.upper()}_mask_original_2D'
    subjects = sorted(os.listdir(folder), key=key_sort_files)
    
    out = ""
    for s,subject in enumerate(subjects):
        
        out += """\\begin{figure}[H]
    \centering\n"""
        out += "	\\includegraphics[page="+str(s+es+1)+",width=\\textwidth]{figures/"+f"{fig_type}"+"_crop.pdf}\n"
        out += "	\caption{"+info_txt+" for subject " + subject + " in the " + dataset_name + " dataset."+extra_caption+"}\n"
        out += "	\\label{fig:"+fig_type+"_" + subject + "}\n"
        out += "\\end{figure}\n\n"
        
    return out



def comp(dataset = "mad_ous"):
    output = get_latex_figures(dataset = dataset, fig_type="overlays", width="0.16", scale="0.35", info_txt="Overlays of predicted segmentations")
    text_file = open(f"latexcode_comp_figs_{dataset}.md", "wt")
    n = text_file.write(output)
    text_file.close()

def comp0(dataset = "mad_ous"):
    output = get_latex_subfigures0(dataset = dataset, width="0.16", scale="0.35", info_txt="Overlays of predicted segmentations")
    text_file = open(f"latexcode_comp_subfigs_{dataset}_v2.md", "wt")
    n = text_file.write(output)
    text_file.close()

# print(get_latex_subfigures0('mad_ous', fig_type = '_mask_overlay'))
# print(get_latex_subfigures0('mesa', fig_type = '_mask_overlay'))
print(get_latex_figures0(dataset = 'mad_ous', fig_type = 'mask_overlays', extra_caption=' The orange box represent the cropped area.'))
print("\n\n")
print(get_latex_figures0(dataset = 'mesa', fig_type = 'mask_overlays', extra_caption=' The orange box represent the cropped area.'))

# comp()
# comp('mesa')
# comp0()
# comp0('mesa')
# print(get_latex_figures0(dataset = 'mad_ous', info_txt="Overlays of predicted segmentations"))
# print("\n\n")
# print(get_latex_figures0(dataset = 'mesa', info_txt="Overlays of predicted segmentations"))




