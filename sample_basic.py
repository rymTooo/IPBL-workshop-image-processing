sum = 0
for i in range(10):
    sum = sum + i
    print(str(i) + ":" + str(sum))

if sum <= 30 :
    print("sum is under 30")
elif sum <= 50 :
    print("sum is between 30 and 50")
else:
    print("sum is over 50")