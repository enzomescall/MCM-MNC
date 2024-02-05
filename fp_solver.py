import numpy as np
import matplotlib.pyplot as plt



x = np.linspace(0, 6, 8)  # Limit x-axis from 0 to 6
y = logistic_function(x, 2.9, 8)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Percent of submersible path found')
plt.title('Sensitivity to sonar range')
plt.grid(True)
plt.show()