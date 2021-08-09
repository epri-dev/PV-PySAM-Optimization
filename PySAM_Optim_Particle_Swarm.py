"""
Copyright (c) 2021, Electric Power Research Institute
 All rights reserved.
 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:
     * Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
     * Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
     * Neither the name of DER-VET nor the names of its contributors
       may be used to endorse or promote products derived from this software
       without specific prior written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import json
import PySAM.Pvsamv1 as Pvsamv1
import PySAM.Grid as Grid
import PySAM.Singleowner as Singleowner
import PySAM.PySSC as pssc

import matplotlib.pyplot as plt
import pandas as pd
import ast
import numpy as np

import operator
import random
import math

from deap import base
from deap import creator
from deap import tools



#%% user inputs
model_filepath = "./Example SAM Model/Exported model/System1.json" #this is the "export to json" file
cost_inputs_file = './Example SAM Model/System1_cost_inputs.xlsx' #this is the export file from SAM imports browser
system_name = 'System1' #this is used to select the correct column in the cost_inputs_file. It should match the name in the tab in SAM


#optimizer parameters - adjust these to desired accuracy (lower learning parameter leads to a more optimized solution but takes more generations to converge)
learn_param = 0.1
optimization_generations = 8

#model free parameters
gcr_min = 0.1
gcr_max = 0.5
ground_clearance_min = 1
ground_clearance_max = 5


#%%system costs - these calculations reproduce the total installed costs on the "System Costs" page of SAM. PySAM does not update total_installed_costs as upstream parameters are changed so it must be done here
#this function assumes the module in SAM is the CEC performance model with user entered specifications. If another module model is used, change the inputs that begin with 'sixpar'
def calc_total_installed_cost(pvsam_model, cost_parameters):
    
    #calculate system inputs
    module_units = (pvsam_model.SystemDesign.subarray1_nstrings*pvsam_model.SystemDesign.subarray1_modules_per_string +
                pvsam_model.SystemDesign.subarray2_nstrings*pvsam_model.SystemDesign.subarray2_modules_per_string*pvsam_model.SystemDesign.subarray2_enable +
                pvsam_model.SystemDesign.subarray3_nstrings*pvsam_model.SystemDesign.subarray3_modules_per_string*pvsam_model.SystemDesign.subarray3_enable +
                pvsam_model.SystemDesign.subarray4_nstrings*pvsam_model.SystemDesign.subarray4_modules_per_string*pvsam_model.SystemDesign.subarray4_enable)
    module_kwdcperunit = pvsam_model.CECPerformanceModelWithUserEnteredSpecifications.export()['sixpar_vmp'] * pvsam_model.CECPerformanceModelWithUserEnteredSpecifications.export()['sixpar_imp']
    module_kwdc = module_units*module_kwdcperunit
    if pvsam_model.Inverter.inverter_model == 0:#cec model
        inv_kwac = pvsam_model.SystemDesign.inverter_count * pvsam_model.Inverter.inv_snl_paco / 1000
    if pvsam_model.Inverter.inverter_model == 1:#datasheet
        inv_kwac = pvsam_model.SystemDesign.inverter_count * pvsam_model.Inverter.inv_ds_paco / 1000
    module_area = pvsam_model.CECPerformanceModelWithUserEnteredSpecifications.export()['sixpar_area']
    land_area = module_area*(pv.SystemDesign.subarray1_nstrings*pv.SystemDesign.subarray1_modules_per_string/pvsam_model.SystemDesign.subarray1_gcr +
                             pv.SystemDesign.subarray2_nstrings*pv.SystemDesign.subarray2_modules_per_string*pvsam_model.SystemDesign.subarray2_enable/pvsam_model.SystemDesign.subarray2_gcr +
                             pv.SystemDesign.subarray3_nstrings*pv.SystemDesign.subarray3_modules_per_string*pvsam_model.SystemDesign.subarray3_enable/pvsam_model.SystemDesign.subarray3_gcr +
                             pv.SystemDesign.subarray4_nstrings*pv.SystemDesign.subarray4_modules_per_string*pvsam_model.SystemDesign.subarray4_enable/pvsam_model.SystemDesign.subarray4_gcr)*0.0002471
    total_module_area = module_area*(pv.SystemDesign.subarray1_nstrings*pv.SystemDesign.subarray1_modules_per_string +
                                     pv.SystemDesign.subarray2_nstrings*pv.SystemDesign.subarray2_modules_per_string*pvsam_model.SystemDesign.subarray2_enable +
                                     pv.SystemDesign.subarray3_nstrings*pv.SystemDesign.subarray3_modules_per_string*pvsam_model.SystemDesign.subarray3_enable +
                                     pv.SystemDesign.subarray4_nstrings*pv.SystemDesign.subarray4_modules_per_string*pvsam_model.SystemDesign.subarray4_enable)
    
    
    #direct capital costs
    direct_module = module_kwdc*cost_parameters['per_module']
    direct_inverter = inv_kwac*cost_parameters['per_inverter']*1000
    direct_bos = cost_parameters['bos_equip_fixed'] + (cost_parameters['bos_equip_perwatt'])*module_kwdc + cost_parameters['bos_equip_perarea']*total_module_area
    direct_install = cost_parameters['install_labor_fixed'] + (cost_parameters['install_labor_perwatt'])*module_kwdc + cost_parameters['install_labor_perarea']*total_module_area
    direct_margin = cost_parameters['install_margin_fixed'] + cost_parameters['install_margin_perwatt']*module_kwdc + cost_parameters['install_margin_perarea']*total_module_area
    direct_contingency = cost_parameters['contingency_percent']*0.01*(direct_module+direct_inverter+direct_bos+direct_install+direct_margin)
    direct_total = direct_contingency+direct_module+direct_inverter+direct_bos+direct_install+direct_margin
    #indirect capital costs
    indirect_permitting = cost_parameters['permitting_percent']/100*direct_total + cost_parameters['permitting_per_watt']*module_kwdc + cost_parameters['permitting_fixed']
    indirect_engineering = cost_parameters['engr_percent']/100*direct_total + cost_parameters['engr_per_watt']*module_kwdc + cost_parameters['engr_fixed']
    indirect_grid_interconnect = cost_parameters['grid_percent']/100*direct_total + cost_parameters['grid_per_watt']*module_kwdc + cost_parameters['grid_fixed']
    indirect_land_purchase = cost_parameters['land_per_acre']*land_area + cost_parameters['land_percent']/100*direct_total + cost_parameters['land_per_watt']*module_kwdc + cost_parameters['land_fixed']
    indirect_land_prep = cost_parameters['landprep_per_acre']*land_area + cost_parameters['landprep_percent']/100*direct_total + cost_parameters['landprep_per_watt']*module_kwdc + cost_parameters['landprep_fixed']
    indirect_land_salestax = cost_parameters['landprep_per_acre']/100*cost_parameters['sales_tax_rate']
    indirect_total = indirect_permitting+indirect_engineering+indirect_grid_interconnect+indirect_land_purchase+indirect_land_prep+indirect_land_salestax
    #total
    total_installed_cost = direct_total+indirect_total
            
    return total_installed_cost


#%%load SAM model and cost parameters
f = open(model_filepath)
dic = json.load(f)

pv_dat = pssc.dict_to_ssc_table(dic, 'Pvsamv1')
grid_dat = pssc.dict_to_ssc_table(dic, 'Grid')
singleowner_dat = pssc.dict_to_ssc_table(dic, 'Singleowner')
f.close()

pv = Pvsamv1.wrap(pv_dat)
grid = Grid.from_existing(pv)
financial = Singleowner.from_existing(grid)

grid.assign(Grid.wrap(grid_dat).export())
financial.assign(Singleowner.wrap(singleowner_dat).export())

#load cost parameters
cost_parameters = pd.read_excel(cost_inputs_file, index_col=0)
cost_parameters = cost_parameters[[system_name]].fillna("")
#convert data type
def convert_type(x):
    try:
        return ast.literal_eval(x.values[0]) #convert string to basic data type
    except: #if the data is a string, pass
        return x.values[0]
cost_parameters = cost_parameters.apply(lambda x: convert_type(x), axis=1).transpose()


    
#%%setup function to optimize and parameter ranges


gcr_list = []
ground_clearance_list = []
lcoe_list = []


def SAM_run(setpoints):
    global pv, grid, financial, LCOE_list, fcn_call_count
    
    #limit the range of the optimization parameters to 0 and 1
    if setpoints[0]<0:
        setpoints[0]=0
    if setpoints[1]<0:
        setpoints[1]=0
    if setpoints[0]>1:
        setpoints[0]=1
    if setpoints[1]>1:
        setpoints[1]=1
    
    #scale parameters (from 0-1 to min-max)
    gcr_iter = setpoints[0]*(gcr_max - gcr_min) + gcr_min
    ground_clearance_iter = setpoints[1]*(ground_clearance_max - ground_clearance_min) + ground_clearance_min
    
    #update the parameters of the PV model
    pv.CECPerformanceModelWithUserEnteredSpecifications.assign({'sixpar_bifacial_ground_clearance_height':ground_clearance_iter})
    pv.SystemDesign.assign({'subarray1_gcr':gcr_iter})
    pv.SystemDesign.assign({'subarray2_gcr':gcr_iter})
    pv.SystemDesign.assign({'subarray3_gcr':gcr_iter})
    pv.SystemDesign.assign({'subarray4_gcr':gcr_iter})
    
    #execute PV model
    pv.execute()
    grid.execute()
    #update capital cost
    financial.SystemCosts.total_installed_cost = calc_total_installed_cost(pv, cost_parameters)
    #update O&M cost
    financial.SystemCosts.om_fixed = ((cost_parameters['om_fixed']),)
    financial.SystemCosts.om_fixed_escal = cost_parameters['om_fixed_escal']
    financial.SystemCosts.om_capacity = (cost_parameters['om_capacity'],)
    financial.SystemCosts.om_capacity_escal = cost_parameters['om_capacity_escal']
    financial.SystemCosts.om_production = (cost_parameters['om_production'],)
    financial.SystemCosts.om_production_escal = cost_parameters['om_production_escal']
    #execute financial model
    financial.execute()
    #save parameters and results
    gcr_list.append(gcr_iter)
    ground_clearance_list.append(ground_clearance_iter)
    lcoe_list.append(financial.Outputs.lcoe_nom)
    fcn_call_count += 1
    #negate the output since the optimizer tries to maximize the result (we want the minimum LCOE)
    print("LCOE " + str(financial.Outputs.lcoe_nom) + " GCR " + str(gcr_iter) + " Clearance " + str(ground_clearance_iter))
    return -1*financial.Outputs.lcoe_nom,

#%%particle swarm - setup and run optimization
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi_local, phi_global, pmin, pmax):
    u1 = (random.uniform(0, phi_local) for _ in range(len(part)))
    u2 = (random.uniform(0, phi_global) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = list(map(operator.add, part, part.speed))
    part = [np.max([pmax, param_iter]) for param_iter in part]
    part = [np.min([pmin, param_iter]) for param_iter in part]

fcn_call_count = 0
param_optimal = []
for i in range(2):#number of optimization parameters
    param_optimal.append(random.random())

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=10, pmin=0, pmax=1, smin=-1*learn_param, smax=learn_param)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi_local=1, phi_global=3, pmin=0, pmax=1)
toolbox.register("evaluate", SAM_run)

pop = toolbox.population(n=10)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

GEN = optimization_generations

best = None
avglist = []
bestlist_swarm = []
fcncalls_swarm = []

for g in range(int(GEN)):
    for part in pop:
        part.fitness.values = toolbox.evaluate(part)
        if not part.best or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if not best or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values
    for part in pop:
        toolbox.update(part, best)

    # Gather all the fitnesses in one list and print the stats
    logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
    print(logbook.stream)
    avglist.append(-1*logbook[-1]['avg'])
    bestlist_swarm.append(-1*SAM_run(best)[0])
    fcncalls_swarm.append(fcn_call_count)
    


plt.figure()
plt.plot(fcncalls_swarm, avglist)
plt.title("Population Average vs Function Calls")
plt.ylabel("Fitness")
plt.xlabel("Function Call Count")
plt.yscale('log')

plt.figure()
plt.plot(fcncalls_swarm, bestlist_swarm)
plt.title("Optimal Solution vs Function Calls")
plt.ylabel("Fitness")
plt.xlabel("Function Call Count")
plt.yscale('log')
        