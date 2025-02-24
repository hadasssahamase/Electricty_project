# Electricty_project
![alt text](<Screenshot 2025-02-24 164518.png>)

## BUSINESS PROBLEM
- Identify and recommend strategies for optimizing electricity consumption and reducing costs for customers across different cities, focusing on appliance usage patterns, company tariffs, and seasonal variations.

## OBJECTIVES
- identify the appliance with the most consumption rate in different seasons
- come up with a model that can  predict electricity bill 
- identify affordable cities to live due to low tariff rate

## DATA
- The above data set has 12 columns
1. Fan - Represents the electricity consumption of fans in a household, business, or location, likely measured in kilowatt-hours (kWh) over a month.
2. Refrigerator - Represents the electricity consumption of refrigerators, likely measured in kWh over a month.
3. AirConditioner - Represents the electricity consumption of air conditioners, typically measured in kWh for a specific month.
4. Television - Represents the electricity consumption of televisions over the given month, measured in kWh.
5. Monitor - Represents the electricity consumption of monitors (e.g., computer monitors) during the specified month, measured in kWh.
6. MotorPump - Represents the electricity consumption of motor pumps (e.g., for water supply, irrigation, or industrial purposes), measured in kWh for the month.
7. Month - Indicates the specific month during which the data was recorded (e.g., January, February).
8. City - Indicates the geographic location (city) where the data was collected.
9. Company - Refers to the electricity supply company or utility provider responsible for the billing and supply of electricity in the area.
10. MonthlyHours - Represents the number of hours appliances or systems were used during the given month.
11. TariffRate - The rate charged by the electricity company, typically expressed as cost per kWh (e.g., $0.12/kWh).
12. ElectricityBill - The total amount charged for electricity usage during the month.

### Data preparation
- The data had no missing values or duplicates.
- checking for outliers.
![alt text](<Screenshot 2025-02-24 165453.png>)

### Featured engineered columns
1. UsageCategory
2. TotalApplianceHours
3. TariffEfficiency
4. MonthlyCostPerHour
5. season
6. appliance_contribution
 
 ## Data Analysis
 ![alt text](<Screenshot 2025-02-24 170322.png>)
- Refrigerator Dominates: The refrigerator has by far the highest average energy consumption among the listed appliances. Its bar is significantly longer than the others, indicating it consumes a disproportionately large amount of energy compared to the rest.
- Fan and Television Moderate: The fan and television have moderate average energy consumption, with the fan consuming slightly more than the television.
- Air Conditioner, Monitor, and Motor Pump Low: The air conditioner, monitor, and motor pump have relatively low average energy consumption compared to the other appliances.  The air conditioner and monitor have particularly low energy usage.
- Motor Pump Data Missing:  The motor pump bar is missing entirely, suggesting that there might be no data available for this appliance's energy consumption, or it might have been unintentionally omitted.

![alt text](<Screenshot 2025-02-24 170738.png>)
- Positive Correlation: There's a clear positive correlation between the tariff rate and the electricity bill. As the tariff rate increases, the electricity bill tends to increase as well. This is an expected and logical relationship – the more expensive each unit of electricity is, the higher the total cost will be.
- Linear Trend: The data points appear to follow a roughly linear trend, suggesting that the relationship between tariff rate and bill is relatively consistent across the range of values.
- Spread and Variation:  While there's a general linear trend, there's also some spread in the data points, indicating that factors other than the tariff rate might be influencing the electricity bill. This spread is visible particularly at higher tariff rates.

- City-Specific Observations:
- Hyderabad: Consistently exhibits high electricity bills across different tariff rates.
- Vadodara, Mumbai, Ratnagiri, New Delhi: Show a similar trend of increasing bills with increasing tariff rates.
- Ahmedabad and Nagpur: Also follow the general trend, but with potentially lower bills compared to the cities mentioned above.
- Chennai, Dahej, Faridabad: Show relatively lower electricity bills across the range of tariff rates.
- Other Cities: The remaining cities (Noida, Kolkata, Pune, Gurgaon, Shimla, Navi Mumbai) also generally follow the increasing trend, but with some variation.

- Further Analysis and Considerations:
- Factors Influencing Spread: The variation in bills at similar tariff rates could be due to factors like:
Electricity Consumption: Different households or businesses in the same city might use different amounts of electricity.
- Usage Patterns: Time of use and peak demand charges can influence the bill.
- Appliance Efficiency: Energy-efficient appliances can reduce electricity consumption.
- Climate: Temperature and weather conditions can affect electricity usage (e.g., for heating or cooling).

## MODELING
- The model that was used is the Gradient boosting model using the following metrics:
1. MAE 16.618176	
2. MSE 442.419932	
3. RMSE 21.033781	
4. MAPE 0.422486	
5. R2 0.99956

![alt text](<Screenshot 2025-02-24 172037.png>)

## RECOMENDATION
1. Companies that want to venture into electric equipments like refrigerator, fan and television should make energy saving equipments that give them a competitive edge over current market.
2. Cities like ratnagiri, shimla, dahej and vadodara have lower tariff rate hence low electricity bill. 
It advisable for companies that have high power consumption to consider the ratnagiri, shimla, dahej and vadodara as their manufacturing site due to the lower tariff rate.
It also recommended for with low wages to consider those cities.
3. For cities like Navi Mumbai, Mumbai and Pune should consider other sources of electricity like solar, geothermal and wind turbines inorder to substitute so as to lower their electricity bills despite the higher tariff rates.

## For more information
- See the full analysis in the Jupyter Notebook or review this presentation. Contact me at :linkedin.com/in/western-onzere-ml17/  and linkedin.com/in/amase-oyakapeli-7848a8343
![alt text](<Screenshot 2025-02-24 173302.png>)