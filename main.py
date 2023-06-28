from utility import *
import time
from openpyxl import load_workbook

start_time = time.time()

pso_iteration = int(input("Please enter PSO iteration:"))

pso_start_iteration = 5
pso_end_iteration = 100

if pso_iteration == 0:
    pso_start_iteration = int(input("Please enter start PSO iteration point:"))
    pso_end_iteration = int(input("Please enter end PSO iteration point:"))

sds_iteration = int(input("Please enter SDS iteration:"))

sds_agent_number = int(input("Please enter SDS agent number:"))

if pso_iteration != 0:
    execute_method(sds_iteration, sds_agent_number, pso_iteration, True)

else:
    sheetNumber = int(sds_iteration / 10) - 1
    wb = load_workbook("Graph.xlsx")
    sheet = wb.worksheets[sheetNumber]

    row_number = 0
    it = 5
    
    for i in range(2, 20):
        value = sheet['D' + str(i)].value
        row_number = i

        if value is None:
            it = sheet['A' + str(i)].value
            if it < pso_start_iteration:
                it = pso_start_iteration
            break

    while it <= pso_end_iteration:
        original_entropy, nuce_entropy, sds_nuce_entropy = execute_method(sds_iteration, sds_agent_number, it, False)

        sheet['B' + str(row_number)].value = original_entropy
        sheet['C' + str(row_number)].value = nuce_entropy
        sheet['D' + str(row_number)].value = sds_nuce_entropy

        wb.save("Graph.xlsx")

        row_number += 1
        it += 5

    print("All iterations finished!")

print("--- %s seconds ---" % (time.time() - start_time))
