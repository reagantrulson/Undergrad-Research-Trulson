import numpy as np
# Rossler system x, a, b, & c = 0,  f(x) - vector field
# what sensor measures s(t) = x(t), or h(x,y,z) = x
x = 1
y = 2
z = 3
# compute lie derivatives L_f^jh(x) - derivative of h along f: h_g * x_dot = d/dt(dh/dx dx/dt) = dh/dx f(x)
# gradient of (h(x) = s(t) = x) = [1,0,0]
# Gradients derived by hand

h_g = [0, 1, 0]
g_g = [1, 0, -0]
i_g = [0,-1,-1]

O = np.array([
    h_g,
    g_g,
    i_g
])

print(O)
print(np.linalg.matrix_rank(O))

# delay embeddings
n = 5000
state_x, state_y = henon(n)
signal = state_x
tau = 1
max_d = 8
scores = []

for d in range(2, max_d + 1):
        E = delay_embedding(signal, tau, d)
        new_delay = E[:, -1]
        previous = E[:, :-1]
        state_trim = np.column_stack((
        state_x[:len(new_delay)],
        state_y[:len(new_delay)]
        ))

        score = pdcor(new_delay, state_trim, previous)
        scores.append(score)
        print(f"d = {d}: {score:.4f}")




