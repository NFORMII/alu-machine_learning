#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, bins=[i for i in range(0, 101, 10)], edgecolour='black')
plt.xticks([i for i in range(0, 101, 10)])
plt.yticks([i for i in range(0, 31, 5)])
plt.xlabel('Grades')
plt.ylabel('Number of students')
plt.title('Project A')
