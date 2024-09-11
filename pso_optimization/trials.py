import numpy 
dim= 3
position = numpy.random.uniform(-1,1,dim)
velocity = numpy.random.uniform(-.5,.5,dim)
# print(position)

# print(velocity)
cognitive_const = 1.5
best_position = position

cognitive_velocity = cognitive_const * numpy.random.random() * (best_position -position)
print(numpy.random.random()) # 0-1 range
print(cognitive_velocity)