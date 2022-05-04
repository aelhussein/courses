# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Part A

total_cost = float(input("What is the cost of your dream home? "))
portion_down_payment = 0.25
downpayment = total_cost*portion_down_payment
annual_salary = float(input("What is your starting salary? "))
monthly_salary = float(annual_salary/12)
portion_saved = float(input("What is the portion of salary to be saved? "))
monthly_salary_saved = monthly_salary * portion_saved
current_savings = 0
r=0.04
months = 0
while current_savings < downpayment:
    current_savings += current_savings*r/12
    current_savings += monthly_salary_saved
    months = months + 1
print("Number of months:", months)

#Part B
total_cost = float(input("What is the cost of your dream home? "))
annual_salary = float(input("What is your starting salary? "))
semi_annual_raise = float(input("what is your semi-annual raise? "))
portion_saved = float(input("What is the portion of salary to be saved? "))
portion_down_payment = 0.25
downpayment = total_cost*portion_down_payment
monthly_salary = float(annual_salary/12)
monthly_salary_saved = monthly_salary * portion_saved
current_savings = 0
r=0.04
months = 0
while current_savings < downpayment:
    current_savings += current_savings*r/12
    current_savings += monthly_salary_saved
    months = months + 1
    if months %  6 == 0:
        annual_salary += annual_salary*semi_annual_raise
        monthly_salary = float(annual_salary/12)
        monthly_salary_saved = monthly_salary * portion_saved
print("Number of months:", months)


#Part C
total_cost = 1000000
annual_salary = float(input("What is your starting salary? "))
semi_annual_raise = 0.07
portion_down_payment = 0.25
downpayment = total_cost*portion_down_payment
low = 0
high = 1000
portion_saved = (low+high)/2000.0
r=0.04
months = 36
epsilon = 100
num_guesses = 0
monthly_salary = float(annual_salary/12)
month = 0
current_savings = 0 
while abs(current_savings-downpayment) >= epsilon:
    monthly_salary = float(annual_salary/12)
    month = 0
    current_savings = 0
    for i in range(months):
        monthly_salary_saved = monthly_salary * portion_saved
        current_savings += current_savings*r/12 + monthly_salary_saved
        month += 1
        if month %  6 == 0:
            monthly_salary += monthly_salary*semi_annual_raise
            monthly_salary_saved = monthly_salary * portion_saved
    if current_savings > downpayment:
        high = portion_saved*1000.0
    else:
        low = portion_saved*1000.0
    portion_saved = (high+low)/2000.0
    num_guesses += 1
    if portion_saved == 1:
        break
if abs(current_savings-downpayment) >= epsilon:
        print("not possible to pay")
else:
    print('Best savings rate', portion_saved)
print("Steps in search", num_guesses)
