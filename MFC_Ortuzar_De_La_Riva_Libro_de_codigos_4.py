#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Analisis del modelo Hibrido, componentes principales


# In[2]:


#Descripcion Matemetica: Formulas utilizadas.


# #Modelo de Black-Scholes:

# In[2]:


import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


# In[4]:


#Simulador.


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from ipywidgets import interact, widgets

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def plot_black_scholes(S, K, T, r, sigma):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    S_range = np.linspace(S - 20, S + 20, 20)
    K_range = np.linspace(K - 20, K + 20, 20)

    X, Y = np.meshgrid(S_range, K_range)
    Z = np.zeros_like(X)

    for i in range(len(S_range)):
        for j in range(len(K_range)):
            Z[i, j] = black_scholes_call(X[i, j], Y[i, j], T, r, sigma)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_title('Black-Scholes Call Price Surface')
    ax.set_xlabel('S')
    ax.set_ylabel('K')
    ax.set_zlabel('Call Price')

    plt.show()

interact(plot_black_scholes, S=(80, 120, 1), K=(80, 120, 1), T=(0.1, 2, 0.1), r=(0.01, 0.1, 0.01), sigma=(0.01, 0.5, 0.01))


# In[ ]:





# In[4]:


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from ipywidgets import interact, widgets

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def interactive_black_scholes(S_min, S_max, volatilidad_min, volatilidad_max):
    # Generate random data for call prices
    S_data = np.random.uniform(S_min, S_max, 1000)
    volatilidad_data = np.random.uniform(volatilidad_min, volatilidad_max, 1000)
    call_prices_data = [black_scholes_call(S, 100, 1, 0.05, volatilidad) for S, volatilidad in zip(S_data, volatilidad_data)]

    # Generate a grid for interpolation
    S_grid, volatilidad_grid = np.meshgrid(np.linspace(S_min, S_max, 100), np.linspace(volatilidad_min, volatilidad_max, 100))

    # Interpolate call prices
    call_prices_smooth = griddata((S_data, volatilidad_data), call_prices_data, (S_grid, volatilidad_grid), method='cubic')

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, volatilidad_grid, call_prices_smooth, cmap='viridis')
    ax.set_xlabel('S')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Superficie Suavizada de Precios de Opción (Black-Scholes)')
    plt.show()

# Create interactive sliders for adjusting parameters
interact(interactive_black_scholes,
         S_min=widgets.FloatSlider(min=80, max=100, step=1, value=80, description='S_min'),
         S_max=widgets.FloatSlider(min=100, max=120, step=1, value=120, description='S_max'),
         volatilidad_min=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.01, description='Volatilidad_min'),
         volatilidad_max=widgets.FloatSlider(min=0.2, max=0.5, step=0.01, value=0.5, description='Volatilidad_max'))


# In[ ]:





# In[5]:


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from ipywidgets import interact, widgets

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def interactive_black_scholes(S_min, S_max, volatilidad_min, volatilidad_max):
    # Generate random data for call prices
    S_data = np.random.uniform(S_min, S_max, 1000)
    volatilidad_data = np.random.uniform(volatilidad_min, volatilidad_max, 1000)
    call_prices_data = [black_scholes_call(S, 100, 1, 0.05, volatilidad) for S, volatilidad in zip(S_data, volatilidad_data)]

    # Generate a grid for interpolation
    S_grid, volatilidad_grid = np.meshgrid(np.linspace(S_min, S_max, 100), np.linspace(volatilidad_min, volatilidad_max, 100))

    # Interpolate call prices
    call_prices_smooth = griddata((S_data, volatilidad_data), call_prices_data, (S_grid, volatilidad_grid), method='cubic')

    # Plot the surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, volatilidad_grid, call_prices_smooth, cmap='viridis')
    
    # Scatter plot for original data points
    ax.scatter(S_data, volatilidad_data, call_prices_data, color='red')

    ax.set_xlabel('S')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Superficie Suavizada de Precios de Opción (Black-Scholes)')
    plt.show()

# Create interactive sliders for adjusting parameters
interact(interactive_black_scholes,
         S_min=widgets.FloatSlider(min=80, max=100, step=1, value=80, description='S_min'),
         S_max=widgets.FloatSlider(min=100, max=120, step=1, value=120, description='S_max'),
         volatilidad_min=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.01, description='Volatilidad_min'),
         volatilidad_max=widgets.FloatSlider(min=0.2, max=0.5, step=0.01, value=0.5, description='Volatilidad_max'))


# In[ ]:





# #Modelo de Heston:

# In[6]:


import numpy as np

def heston_call(S, K, T, r, v0, theta, kappa, sigma, rho, n_simulations=10000, n_steps=252):
    dt = T / n_steps
    call_prices = np.zeros(n_simulations)

    for i in range(n_simulations):
        St = S
        vt = np.zeros(n_steps)
        vt[0] = v0

        for j in range(1, n_steps):
            Z1 = np.random.normal()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal()

            dW1 = np.sqrt(dt) * Z1
            dW2 = np.sqrt(dt) * Z2

            dSt = r * St * dt + np.sqrt(vt[j-1]) * St * dW1
            dvt = kappa * (theta - vt[j-1]) * dt + sigma * np.sqrt(vt[j-1]) * dW2

            St += dSt
            vt[j] = vt[j-1] + dvt

        call_price = max(St - K, 0) * np.exp(-r * T)
        call_prices[i] = call_price

    return np.mean(call_prices)


# In[9]:


#Simulador.


# In[ ]:





# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.integrate import quad
from ipywidgets import interact, widgets

def heston_call(S, K, T, r, v0, theta, kappa, sigma, rho):
    # Parámetros de Heston
    kappa = kappa
    theta = theta
    sigma = sigma
    rho = rho

    # Fórmula de Heston para el precio de la opción de compra
    def integrand(u):
        x = np.log(S)
        a = kappa * theta
        b = -kappa - rho * sigma
        c = sigma**2 / 2

        d = np.sqrt((rho * sigma * 1j * u - b)**2 - sigma**2 * (2 * c * 1j * u - sigma))
        g = (b - rho * sigma * 1j * u + d) / (b - rho * sigma * 1j * u - d)

        C = r * 1j * u * T + a / sigma**2 * ((b - rho * sigma * 1j * u + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))

        D = (b - rho * sigma * 1j * u + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

        return np.exp(C + D * v0 - 1j * u * np.log(K)) / (1j * u)

    # Manejar posibles valores complejos o infinitos
    try:
        integral_result, _ = quad(integrand, 0, 100)  # Ajuste de límites de integración
    except (ValueError, ZeroDivisionError):
        return np.nan

    # Extraer la parte real de los resultados de la integración
    integral_result_real = np.real(integral_result)

    # Fórmula final de Heston
    call_price = S * np.exp(-r * T) * (0.5 + 1/np.pi * integral_result_real)

    return call_price if np.isreal(call_price) and not np.isinf(call_price) else np.nan

def plot_heston(S, K, T, r, v0, theta, kappa, sigma, rho):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    S_range = np.linspace(80, 120, 20)
    K_range = np.linspace(80, 120, 20)

    X, Y = np.meshgrid(S_range, K_range)
    Z = np.zeros_like(X)

    for i in range(len(S_range)):
        for j in range(len(K_range)):
            Z[i, j] = heston_call(X[i, j], Y[i, j], T, r, v0, theta, kappa, sigma, rho)

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('Heston Call Price Surface')
    ax.set_xlabel('S')
    ax.set_ylabel('K')
    ax.set_zlabel('Call Price')

    plt.show()

interact(plot_heston, S=(80, 120, 1), K=(80, 120, 1), T=(0.1, 2, 0.1), r=(0.01, 0.1, 0.01), v0=(0.01, 0.5, 0.01), theta=(0.01, 0.5, 0.01), kappa=(0.1, 5, 0.1), sigma=(0.01, 0.5, 0.01), rho=(-0.9, 0.9, 0.1))


# In[ ]:





# In[9]:


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, widgets

def heston_call(S, K, T, r, v0, theta, kappa, sigma, rho):
    # Parámetros de Heston
    kappa = kappa
    theta = theta
    sigma = sigma
    rho = rho

    # Fórmula de Heston para el precio de la opción de compra
    def integrand(u):
        x = np.log(S)
        a = kappa * theta
        b = -kappa - rho * sigma
        c = sigma**2 / 2

        d = np.sqrt((rho * sigma * 1j * u - b)**2 - sigma**2 * (2 * c * 1j * u - sigma))
        g = (b - rho * sigma * 1j * u + d) / (b - rho * sigma * 1j * u - d)

        C = r * 1j * u * T + a / sigma**2 * ((b - rho * sigma * 1j * u + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))

        D = (b - rho * sigma * 1j * u + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

        return np.exp(C + D * v0 - 1j * u * np.log(K)) / (1j * u)

    # Manejar posibles valores complejos o infinitos
    try:
        integral_result, _ = quad(integrand, 0, 100)  # Ajuste de límites de integración
    except (ValueError, ZeroDivisionError):
        return np.nan

    # Extraer la parte real de los resultados de la integración
    integral_result_real = np.real(integral_result)

    # Fórmula final de Heston
    call_price = S * np.exp(-r * T) * (0.5 + 1/np.pi * integral_result_real)

    return call_price if np.isreal(call_price) and not np.isinf(call_price) else np.nan

def smooth_smile(S_min, S_max, vol_min, vol_max, K, T, r, v0, theta, kappa, sigma, rho):
    # Generar datos para la sonrisa de volatilidad utilizando Heston
    S_range = np.linspace(S_min, S_max, 50)
    volatilidad = np.linspace(vol_min, vol_max, 50)
    S_grid, volatilidad_grid = np.meshgrid(S_range, volatilidad)
    call_prices_smooth = np.zeros_like(S_grid)

    for i in range(len(S_range)):
        for j in range(len(volatilidad)):
            call_prices_smooth[i, j] = heston_call(S_grid[i, j], K, T, r, v0, theta, kappa, sigma, rho)

    # Interpolar los datos para obtener una superficie suavizada
    S_flat = S_grid.flatten()
    volatilidad_flat = volatilidad_grid.flatten()
    points = np.column_stack((S_flat, volatilidad_flat))
    values = call_prices_smooth.flatten()
    grid_x, grid_y = np.meshgrid(np.linspace(S_min, S_max, 50), np.linspace(vol_min, vol_max, 50))
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    # Graficar la superficie suavizada
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie con trama de alambre y transparencia
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.5, edgecolor='none')

    # Líneas de contorno en la proyección XY
    ax.contour(grid_x, grid_y, grid_z, zdir='z', offset=0, cmap='viridis')

    ax.set_xlabel('S')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Superficie de la Sonrisa de Volatilidad Suavizada')

    plt.show()

# Controles deslizantes interactivos para ajustar los rangos de S y volatilidad
interact(smooth_smile,
         S_min=widgets.FloatSlider(min=80, max=100, step=1, value=80, description='S_min'),
         S_max=widgets.FloatSlider(min=100, max=120, step=1, value=120, description='S_max'),
         vol_min=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.01, description='Vol_min'),
         vol_max=widgets.FloatSlider(min=0.2, max=0.5, step=0.01, value=0.5, description='Vol_max'),
         K=widgets.FloatSlider(min=80, max=120, step=1, value=100, description='K'),
         T=widgets.FloatSlider(min=0.1, max=2, step=0.1, value=1, description='T'),
         r=widgets.FloatSlider(min=0.01, max=0.1, step=0.01, value=0.05, description='r'),
         v0=widgets.FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='v0'),
         theta=widgets.FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='theta'),
         kappa=widgets.FloatSlider(min=0.1, max=5, step=0.1, value=2, description='kappa'),
         sigma=widgets.FloatSlider(min=0.01, max=0.5, step=0.01, value=0.2, description='sigma'),
         rho=widgets.FloatSlider(min=-0.9, max=0.9, step=0.1, value=0.1, description='rho'))


# In[ ]:





# In[10]:


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.integrate import quad
from ipywidgets import interact, widgets

def heston_call(S, K, T, r, v0, theta, kappa, sigma, rho):
    # Parámetros de Heston
    kappa = kappa
    theta = theta
    sigma = sigma
    rho = rho

    # Fórmula de Heston para el precio de la opción de compra
    def integrand(u):
        x = np.log(S)
        a = kappa * theta
        b = -kappa - rho * sigma
        c = sigma**2 / 2

        d = np.sqrt((rho * sigma * 1j * u - b)**2 - sigma**2 * (2 * c * 1j * u - sigma))
        g = (b - rho * sigma * 1j * u + d) / (b - rho * sigma * 1j * u - d)

        C = r * 1j * u * T + a / sigma**2 * ((b - rho * sigma * 1j * u + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))

        D = (b - rho * sigma * 1j * u + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

        return np.exp(C + D * v0 - 1j * u * np.log(K)) / (1j * u)

    # Manejar posibles valores complejos o infinitos
    try:
        integral_result, _ = quad(integrand, 0, 100)  # Ajuste de límites de integración
    except (ValueError, ZeroDivisionError):
        return np.nan

    # Extraer la parte real de los resultados de la integración
    integral_result_real = np.real(integral_result)

    # Fórmula final de Heston
    call_price = S * np.exp(-r * T) * (0.5 + 1/np.pi * integral_result_real)

    return call_price if np.isreal(call_price) and not np.isinf(call_price) else np.nan

def smooth_smile(S_min, S_max, vol_min, vol_max, K, T, r, v0, theta, kappa, sigma, rho):
    # Generar datos para la sonrisa de volatilidad utilizando Heston
    S_range = np.linspace(S_min, S_max, 50)
    volatilidad = np.linspace(vol_min, vol_max, 50)
    S_grid, volatilidad_grid = np.meshgrid(S_range, volatilidad)
    call_prices_smooth = np.zeros_like(S_grid)

    for i in range(len(S_range)):
        for j in range(len(volatilidad)):
            call_prices_smooth[i, j] = heston_call(S_grid[i, j], K, T, r, v0, theta, kappa, sigma, rho)

    # Interpolar los datos para obtener una superficie suavizada
    S_flat = S_grid.flatten()
    volatilidad_flat = volatilidad_grid.flatten()
    points = np.column_stack((S_flat, volatilidad_flat))
    values = call_prices_smooth.flatten()
    grid_x, grid_y = np.meshgrid(np.linspace(S_min, S_max, 50), np.linspace(vol_min, vol_max, 50))
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    # Graficar la superficie suavizada
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie con trama de alambre y transparencia
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.5, edgecolor='none')

    # Líneas de contorno en la proyección XY
    ax.contour(grid_x, grid_y, grid_z, zdir='z', offset=0, cmap='viridis')

    # Gráfico de dispersión de los datos originales
    ax.scatter(S_flat, volatilidad_flat, values, color='red', alpha=0.6)

    ax.set_xlabel('S')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Superficie de la Sonrisa de Volatilidad Suavizada')

    plt.show()

# Controles deslizantes interactivos para ajustar los rangos de S y volatilidad
interact(smooth_smile,
         S_min=widgets.FloatSlider(min=80, max=100, step=1, value=80, description='S_min'),
         S_max=widgets.FloatSlider(min=100, max=120, step=1, value=120, description='S_max'),
         vol_min=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.01, description='Vol_min'),
         vol_max=widgets.FloatSlider(min=0.2, max=0.5, step=0.01, value=0.5, description='Vol_max'),
         K=widgets.FloatSlider(min=80, max=120, step=1, value=100, description='K'),
         T=widgets.FloatSlider(min=0.1, max=2, step=0.1, value=1, description='T'),
         r=widgets.FloatSlider(min=0.01, max=0.1, step=0.01, value=0.05, description='r'),
         v0=widgets.FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='v0'),
         theta=widgets.FloatSlider(min=0.01, max=0.5, step=0.01, value=0.1, description='theta'),
         kappa=widgets.FloatSlider(min=0.1, max=5, step=0.1, value=2, description='kappa'),
         sigma=widgets.FloatSlider(min=0.01, max=0.5, step=0.01, value=0.2, description='sigma'),
         rho=widgets.FloatSlider(min=-0.9, max=0.9, step=0.1, value=0.1, description='rho'))


# In[ ]:





# #Modelo Híbrido Tradicional Black-Scholes-Heston:

# In[11]:


def hybrid_model_bs_heston(S, K, T, r, v0, theta, kappa, sigma, rho, n_simulations=10000, n_steps=252):
    dt = T / n_steps
    call_prices = np.zeros(n_simulations)

    for i in range(n_simulations):
        St = S
        vt = np.zeros(n_steps)
        vt[0] = v0

        for j in range(1, n_steps):
            Z1 = np.random.normal()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal()

            dW1 = np.sqrt(dt) * Z1
            dW2 = np.sqrt(dt) * Z2

            dSt_bs = r * St * dt + np.sqrt(vt[j-1]) * St * dW1
            dvt_heston = kappa * (theta - vt[j-1]) * dt + sigma * np.sqrt(vt[j-1]) * dW2

            St += dSt_bs
            vt[j] = vt[j-1] + dvt_heston

        call_price = max(St - K, 0) * np.exp(-r * T)
        call_prices[i] = call_price

    return np.mean(call_prices)


# In[14]:


#Simulador.


# In[ ]:





# In[12]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.integrate import quad
from ipywidgets import interact, widgets

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def heston_call(S, K, T, r, v0, theta, kappa, sigma, rho):
    # Parámetros de Heston
    kappa = kappa
    theta = theta
    sigma = sigma
    rho = rho

    # Fórmula de Heston para el precio de la opción de compra
    def integrand(u):
        x = np.log(S)
        a = kappa * theta
        b = -kappa - rho * sigma
        c = sigma**2 / 2

        d = np.sqrt((rho * sigma * 1j * u - b)**2 - sigma**2 * (2 * c * 1j * u - sigma))
        g = (b - rho * sigma * 1j * u + d) / (b - rho * sigma * 1j * u - d)

        C = r * 1j * u * T + a / sigma**2 * ((b - rho * sigma * 1j * u + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))

        D = (b - rho * sigma * 1j * u + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

        return np.exp(C + D * v0 - 1j * u * np.log(K)) / (1j * u)

    integral_result, _ = quad(integrand, 0, np.inf)

    # Fórmula final de Heston
    call_price = S * np.exp(-r * T) * (0.5 + 1/np.pi * integral_result)

    return call_price

def hybrid_model_bs_heston(S, K, T, r, v0, theta, kappa, sigma, rho):
    # Asumiendo que la volatilidad de Black-Scholes es constante durante el tiempo T
    bs_price = black_scholes_call(S, K, T, r, sigma)

    # Asumiendo que la volatilidad de Heston se mantiene constante durante el tiempo T
    heston_price = heston_call(S, K, T, r, v0, theta, kappa, sigma, rho)

    # Mezcla lineal de los precios de Black-Scholes y Heston
    hybrid_price = 0.5 * bs_price + 0.5 * heston_price

    return hybrid_price

def plot_hybrid_bs_heston(S, K, T, r, v0, theta, kappa, sigma, rho):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    S_range = np.linspace(80, 120, 20)
    K_range = np.linspace(80, 120, 20)

    X, Y = np.meshgrid(S_range, K_range)
    Z = np.zeros_like(X)

    for i in range(len(S_range)):
        for j in range(len(K_range)):
            Z[i, j] = hybrid_model_bs_heston(S_range[i], K_range[j], T, r, v0, theta, kappa, sigma, rho)

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title('Hybrid BS-Heston Call Price Surface')
    ax.set_xlabel('S')
    ax.set_ylabel('K')
    ax.set_zlabel('Call Price')

    plt.show()

interact(plot_hybrid_bs_heston, S=(80, 120, 1), K=(80, 120, 1), T=(0.1, 2, 0.1), r=(0.01, 0.1, 0.01), v0=(0.01, 0.5, 0.01), theta=(0.01, 0.5, 0.01), kappa=(0.1, 5, 0.1), sigma=(0.01, 0.5, 0.01), rho=(-0.9, 0.9, 0.1))


# In[ ]:





# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, widgets

def hybrid_model_bs_heston(S, K, T, r, v0, theta, kappa, sigma, rho):
    # Asumiendo que la volatilidad de Black-Scholes es constante durante el tiempo T
    bs_price = black_scholes_call(S, K, T, r, sigma)

    # Asumiendo que la volatilidad de Heston se mantiene constante durante el tiempo T
    heston_price = heston_call(S, K, T, r, v0, theta, kappa, sigma, rho)

    # Mezcla lineal de los precios de Black-Scholes y Heston
    hybrid_price = 0.5 * bs_price + 0.5 * heston_price

    return hybrid_price

def smooth_smile_hybrid(S_min, S_max, vol_min, vol_max):
    # Generar datos de ejemplo para la sonrisa de volatilidad
    S = np.linspace(S_min, S_max, 50)
    volatilidad = np.linspace(vol_min, vol_max, 50)
    S_grid, volatilidad_grid = np.meshgrid(S, volatilidad)
    call_prices_smooth = np.random.rand(50, 50)  # Reemplazar con los datos reales

    # Interpolar los datos para obtener una superficie suavizada
    S_flat = S_grid.flatten()
    volatilidad_flat = volatilidad_grid.flatten()
    points = np.column_stack((S_flat, volatilidad_flat))
    values = call_prices_smooth.flatten()
    grid_x, grid_y = np.meshgrid(np.linspace(S_min, S_max, 50), np.linspace(vol_min, vol_max, 50))
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    # Graficar la superficie suavizada
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie con trama de alambre y transparencia
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.5, edgecolor='none')

    # Líneas de contorno en la proyección XY
    ax.contour(grid_x, grid_y, grid_z, zdir='z', offset=0, cmap='viridis')

    ax.set_xlabel('S')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Superficie de la Sonrisa de Volatilidad Suavizada (Modelo Híbrido)')

    plt.show()

# Crear controles deslizantes interactivos para ajustar los rangos de S y volatilidad
interact(smooth_smile_hybrid,
         S_min=widgets.FloatSlider(min=80, max=100, step=1, value=80, description='S_min'),
         S_max=widgets.FloatSlider(min=100, max=120, step=1, value=120, description='S_max'),
         vol_min=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.01, description='Vol_min'),
         vol_max=widgets.FloatSlider(min=0.2, max=0.5, step=0.01, value=0.5, description='Vol_max'))


# In[ ]:





# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from ipywidgets import interact, widgets

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def heston_call(S, K, T, r, v0, theta, kappa, sigma, rho):
    # Parámetros de Heston
    kappa = kappa
    theta = theta
    sigma = sigma
    rho = rho

    # Fórmula de Heston para el precio de la opción de compra
    def integrand(u):
        x = np.log(S)
        a = kappa * theta
        b = -kappa - rho * sigma
        c = sigma**2 / 2

        d = np.sqrt((rho * sigma * 1j * u - b)**2 - sigma**2 * (2 * c * 1j * u - sigma))
        g = (b - rho * sigma * 1j * u + d) / (b - rho * sigma * 1j * u - d)

        C = r * 1j * u * T + a / sigma**2 * ((b - rho * sigma * 1j * u + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))

        D = (b - rho * sigma * 1j * u + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

        return np.exp(C + D * v0 - 1j * u * np.log(K)) / (1j * u)

    integral_result, _ = quad(integrand, 0, np.inf)

    # Fórmula final de Heston
    call_price = S * np.exp(-r * T) * (0.5 + 1/np.pi * integral_result)

    return call_price

def hybrid_model_bs_heston(S, K, T, r, v0, theta, kappa, sigma, rho):
    # Asumiendo que la volatilidad de Black-Scholes es constante durante el tiempo T
    bs_price = black_scholes_call(S, K, T, r, sigma)

    # Asumiendo que la volatilidad de Heston se mantiene constante durante el tiempo T
    heston_price = heston_call(S, K, T, r, v0, theta, kappa, sigma, rho)

    # Mezcla lineal de los precios de Black-Scholes y Heston
    hybrid_price = 0.5 * bs_price + 0.5 * heston_price

    return hybrid_price

def smooth_smile_hybrid(S_min, S_max, vol_min, vol_max):
    # Generar datos de ejemplo para la sonrisa de volatilidad
    S = np.linspace(S_min, S_max, 50)
    volatilidad = np.linspace(vol_min, vol_max, 50)
    S_grid, volatilidad_grid = np.meshgrid(S, volatilidad)
    call_prices_smooth = np.random.rand(50, 50)  # Reemplazar con los datos reales

    # Interpolar los datos para obtener una superficie suavizada
    S_flat = S_grid.flatten()
    volatilidad_flat = volatilidad_grid.flatten()
    points = np.column_stack((S_flat, volatilidad_flat))
    values = call_prices_smooth.flatten()
    grid_x, grid_y = np.meshgrid(np.linspace(S_min, S_max, 50), np.linspace(vol_min, vol_max, 50))
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    # Graficar la superficie suavizada
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Superficie con trama de alambre y transparencia
    ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', alpha=0.5, edgecolor='none')

    # Líneas de contorno en la proyección XY
    ax.contour(grid_x, grid_y, grid_z, zdir='z', offset=0, cmap='viridis')

    # Gráfico de dispersión de los datos originales
    ax.scatter(S_flat, volatilidad_flat, values, color='red', alpha=0.6)

    ax.set_xlabel('S')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Superficie de la Sonrisa de Volatilidad Suavizada (Modelo Híbrido)')

    plt.show()

# Crear controles deslizantes interactivos para ajustar los rangos de S y volatilidad
interact(smooth_smile_hybrid,
         S_min=widgets.FloatSlider(min=80, max=100, step=1, value=80, description='S_min'),
         S_max=widgets.FloatSlider(min=100, max=120, step=1, value=120, description='S_max'),
         vol_min=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.01, description='Vol_min'),
         vol_max=widgets.FloatSlider(min=0.2, max=0.5, step=0.01, value=0.5, description='Vol_max'))


# In[ ]:





# #Modelo Híbrido LSTM-Heston con componentes de redes neuronales:

# In[16]:


import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_lstm_model(input_shape=(10, 1)):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def hybrid_model_lstm_heston(S, K, T, r, lstm_model, volatilidad_inicial, theta, kappa, sigma, n_simulations=10000, n_steps=252):
    dt = T / n_steps
    S_t = S
    vt = np.zeros(n_steps + 1) + volatilidad_inicial
    rand = np.random.normal(size=n_steps)

    call_prices = np.zeros(n_simulations)

    for sim in range(n_simulations):
        for i in range(1, n_steps + 1):
            vt[i] = (vt[i - 1] +
                     kappa * (theta - vt[i - 1]) * dt +
                     sigma * np.sqrt(vt[i - 1]) * rand[i - 1])

        X_lstm = np.zeros((1, 10, 1))
        X_lstm[:, :, 0] = vt[-10:]

        lstm_output = lstm_model.predict(X_lstm)

        call_prices[sim] = np.maximum(S_t - K, 0) * np.exp(-r * T) + lstm_output.flatten()

    return np.mean(call_prices)


# In[19]:


pip install keras


# In[20]:


#Simulador.


# In[ ]:





# In[17]:


import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, widgets, fixed

# Función para construir el modelo LSTM
def build_lstm_model(input_shape=(10, 1)):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Función para el modelo híbrido LSTM-Heston
def hybrid_model_lstm_heston(S, K, T, r, lstm_model, volatilidad_inicial, theta, kappa, sigma, n_simulations=1000, n_steps=252):
    # Lógica del modelo híbrido LSTM-Heston
    dt = T / n_steps
    S_t = S
    vt = np.zeros((n_simulations, n_steps + 1))  # Corregir el tamaño de vt
    rand = np.random.normal(size=(n_simulations, n_steps))

    call_prices = np.zeros(n_simulations)

    for sim in range(n_simulations):
        for i in range(1, n_steps + 1):
            vt[sim, i] = (vt[sim, i - 1] +
                     kappa * (theta - vt[sim, i - 1]) * dt +
                     sigma * np.sqrt(abs(vt[sim, i - 1])) * rand[sim, i - 1])

        X_lstm = np.zeros((1, 10, 1))
        X_lstm[:, :, 0] = vt[sim, -10:]  # Seleccionar los últimos 10 valores de vt

        lstm_output = lstm_model.predict(X_lstm)

        call_prices[sim] = np.maximum(S_t - K, 0) * np.exp(-r * T) + lstm_output.flatten()

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.arange(n_simulations), vt[:, -1], call_prices, c='b', marker='o')
    ax.set_xlabel('Simulaciones')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Distribución de Precios de la Opción en 3D')
    plt.show()

# Datos de ejemplo para X_train e y_train (ajusta según tus datos)
# Asegúrate de tener datos de entrada y salida apropiados para tu problema
X_train = np.random.rand(100, 10, 1)
y_train = np.random.rand(100, 1)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Modelar y entrenar el modelo LSTM
lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
lstm_model.save('tu_modelo_lstm.h5')

# Simulador interactivo
interact(hybrid_model_lstm_heston, S=(80, 120, 1), K=(80, 120, 1), T=(0.1, 2, 0.1), r=(0.01, 0.1, 0.01), volatilidad_inicial=(0.01, 0.5, 0.01), theta=(0.01, 0.5, 0.01), kappa=(0.1, 5, 0.1), sigma=(0.01, 0.5, 0.01), lstm_model=fixed(lstm_model))


# In[ ]:





# In[21]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, widgets, fixed
from keras.models import load_model

def build_lstm_model(input_shape=(10, 1)):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def hybrid_model_lstm_heston(S, K, T, r, lstm_model, volatilidad_inicial, theta, kappa, sigma, n_simulations=100, n_steps=252):
    dt = T / n_steps
    S_t = S
    vt = np.zeros(n_steps + 1) + volatilidad_inicial
    rand = np.random.normal(size=n_steps)

    call_prices = np.zeros(n_simulations)

    for sim in range(n_simulations):
        for i in range(1, n_steps + 1):
            vt[i] = (vt[i - 1] +
                     kappa * (theta - vt[i - 1]) * dt +
                     sigma * np.sqrt(vt[i - 1]) * rand[i - 1])

        X_lstm = np.zeros((1, 10, 1))
        X_lstm[:, :, 0] = vt[-10:]

        lstm_output = lstm_model.predict(X_lstm)

        call_prices[sim] = np.maximum(S_t - K, 0) * np.exp(-r * T) + lstm_output.flatten()

    return np.mean(call_prices)

def hybrid_model_lstm_heston_call_surface(S_range, K_range, T, r, lstm_model, volatilidad_inicial, theta, kappa, sigma, rho):
    call_prices_surface = np.zeros((len(S_range), len(K_range)))

    for i, S in enumerate(S_range):
        for j, K in enumerate(K_range):
            call_price = hybrid_model_lstm_heston(S, K, T, r, lstm_model, volatilidad_inicial, theta, kappa, sigma)
            call_prices_surface[i, j] = call_price

    S_grid, K_grid = np.meshgrid(S_range, K_range)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, K_grid, call_prices_surface, cmap='viridis')
    ax.set_xlabel('S')
    ax.set_ylabel('K')
    ax.set_zlabel('Call Price')
    ax.set_title('Hybrid LSTM-Heston Call Price Surface')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# Cargar el modelo LSTM entrenado
lstm_model = load_model('tu_modelo_lstm.h5')

# Definir los rangos para S y K
S_range = np.linspace(80, 120, 20)
K_range = np.linspace(80, 120, 20)

# Crear controles deslizantes interactivos para ajustar los parámetros
interact(hybrid_model_lstm_heston_call_surface, S_range=fixed(S_range), K_range=fixed(K_range), T=(0.1, 2, 0.1), r=(0.01, 0.1, 0.01), lstm_model=fixed(lstm_model), volatilidad_inicial=(0.01, 0.5, 0.01), theta=(0.01, 0.5, 0.01), kappa=(0.1, 5, 0.1), sigma=(0.01, 0.5, 0.01), rho=(-0.9, 0.9, 0.1))


# In[ ]:





# In[19]:


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from keras.layers import LSTM
from ipywidgets import interact, widgets, fixed

def interactive_hybrid_plot(S_min, S_max, vol_min, vol_max, lstm_model):
    # Supongamos que tienes tus datos en las siguientes matrices
    S = np.random.uniform(S_min, S_max, 1000)
    volatilidad = np.random.uniform(vol_min, vol_max, 1000)
    call_prices = np.random.rand(1000)

    # Generar una malla para la interpolación
    S_grid, vol_grid = np.meshgrid(np.linspace(S.min(), S.max(), 100),
                                   np.linspace(volatilidad.min(), volatilidad.max(), 100))

    # Interpolar los precios de las opciones
    call_prices_smooth = griddata((S, volatilidad), call_prices, (S_grid, vol_grid), method='cubic')

    # Modelar y predecir con LSTM
    lstm_output = lstm_model.predict(np.random.rand(1, 10, 1))  # Insertar datos reales aquí

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, vol_grid, call_prices_smooth, cmap='viridis', alpha=0.5)
    ax.scatter(S_max, vol_max, lstm_output, c='r', marker='o')  # Punto de predicción LSTM
    ax.set_xlabel('S')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Distribución de Precios de la Opción Suavizada con Predicción LSTM')
    plt.show()

# Cargar el modelo LSTM entrenado
lstm_model = load_model('tu_modelo_lstm.h5')

# Crear controles deslizantes interactivos para ajustar los rangos de S y volatilidad
interact(interactive_hybrid_plot,
         S_min=widgets.FloatSlider(min=80, max=100, step=1, value=80, description='S_min'),
         S_max=widgets.FloatSlider(min=100, max=120, step=1, value=120, description='S_max'),
         vol_min=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.01, description='Vol_min'),
         vol_max=widgets.FloatSlider(min=0.2, max=0.5, step=0.01, value=0.5, description='Vol_max'),
         lstm_model=fixed(lstm_model))


# In[ ]:





# In[20]:


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from keras.layers import LSTM
from ipywidgets import interact, widgets, fixed

def interactive_hybrid_plot(S_min, S_max, vol_min, vol_max, lstm_model):
    # Supongamos que tienes tus datos en las siguientes matrices
    S = np.random.uniform(S_min, S_max, 1000)
    volatilidad = np.random.uniform(vol_min, vol_max, 1000)
    call_prices = np.random.rand(1000)

    # Generar una malla para la interpolación
    S_grid, vol_grid = np.meshgrid(np.linspace(S.min(), S.max(), 100),
                                   np.linspace(volatilidad.min(), volatilidad.max(), 100))

    # Interpolar los precios de las opciones
    call_prices_smooth = griddata((S, volatilidad), call_prices, (S_grid, vol_grid), method='cubic')

    # Modelar y predecir con LSTM
    lstm_output = lstm_model.predict(np.random.rand(1, 10, 1))  # Insertar datos reales aquí

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S_grid, vol_grid, call_prices_smooth, cmap='viridis', alpha=0.5)
    ax.scatter(S, volatilidad, call_prices, c='r', marker='o')  # Puntos rojos
    ax.set_xlabel('S')
    ax.set_ylabel('Volatilidad')
    ax.set_zlabel('Precio de la Opción')
    ax.set_title('Distribución de Precios de la Opción Suavizada con Predicción LSTM')
    plt.show()

# Cargar el modelo LSTM entrenado
lstm_model = load_model('tu_modelo_lstm.h5')

# Crear controles deslizantes interactivos para ajustar los rangos de S y volatilidad
interact(interactive_hybrid_plot,
         S_min=widgets.FloatSlider(min=80, max=100, step=1, value=80, description='S_min'),
         S_max=widgets.FloatSlider(min=100, max=120, step=1, value=120, description='S_max'),
         vol_min=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.01, description='Vol_min'),
         vol_max=widgets.FloatSlider(min=0.2, max=0.5, step=0.01, value=0.5, description='Vol_max'),
         lstm_model=fixed(lstm_model))


# In[ ]:





# Descripcion mateematica de los modelos utilizados en la simulacion (Memoria+Modelos+Simulacion).

# Descripción del componente LSTM y Heston del modelo Hibrido:
# 
# Componente LSTM:
# 
# El componente LSTM es una red neuronal recurrente que se utiliza para modelar secuencias de datos con dependencias temporales.
# En el contexto del modelo híbrido, el componente LSTM se encarga de procesar y modelar la secuencia temporal de datos históricos relevantes para la valoración de opciones, como los precios de mercado, volúmenes de operación y otros factores.
# La red LSTM aprende patrones complejos en los datos históricos y captura relaciones no lineales que pueden ser útiles para predecir futuros movimientos del mercado, incluyendo cambios en la volatilidad implícita.
# Después de entrenado, el componente LSTM proporciona proyecciones de futuros precios o volatilidades, que se utilizan como entrada en el modelo híbrido para mejorar la precisión de la valoración de opciones.
# 
# Componente Heston:
# 
# El componente Heston es un modelo estocástico utilizado para modelar la dinámica de la volatilidad implícita en el mercado financiero.
# En el contexto del modelo híbrido, el componente Heston se utiliza para modelar la evolución estocástica de la volatilidad implícita a lo largo del tiempo.
# El modelo de Heston introduce un proceso estocástico para la volatilidad, lo que permite capturar la sonrisa de volatilidad y otros patrones observados empíricamente en los mercados financieros.
# Al integrar el componente Heston en el modelo híbrido, se mejora la capacidad del modelo para capturar la dinámica compleja de la volatilidad implícita y proporcionar valoraciones más precisas de las opciones financieras.
# En resumen, el componente LSTM se centra en modelar patrones en los datos históricos, mientras que el componente Heston se enfoca en modelar la dinámica estocástica de la volatilidad implícita. Ambos componentes se integran en el modelo híbrido para mejorar la precisión de la valoración de opciones financieras.
# 
# 
# 
# Lenguaje de código:
# 
# Black-Scholes:
# 
# # Función para calcular el precio de una opción de compra con Black-Scholes
# def black_scholes_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#     return call_price
# 
# Heston:
# # Función para calcular el precio de una opción de compra con Heston (volatilidad no constante)
# def heston_call_time_varying_volatility(S, K, T, r, v0, theta, kappa, sigma, n_simulations=10000, n_steps=252):
#     dt = T / n_steps
#     call_prices = np.zeros(n_simulations)
# 
#     for i in range(n_simulations):
#         St = S
#         vt = np.zeros(n_steps)
#         vt[0] = v0
# 
#         for j in range(1, n_steps):
#             Z1 = np.random.normal()
#             Z2 = np.random.normal()
# 
#             dW1 = np.sqrt(dt) * Z1
#             dW2 = np.sqrt(dt) * Z2
# 
#             dSt = r * St * dt + np.sqrt(vt[j-1]) * St * dW1
#             dvt = kappa * (theta - vt[j-1]) * dt + sigma * np.sqrt(vt[j-1]) * dW2
# 
#             St += dSt
#             vt[j] = vt[j-1] + dvt
# 
#         call_price = max(St - K, 0) * np.exp(-r * T)
#         call_prices[i] = call_price
# 
#     return np.mean(call_prices)
# 
# Modelo Híbrido tradicional Black-Sholes-Heston:
# # Función para calcular el precio de una opción de compra con el modelo híbrido
# def hybrid_model_call(S, K, T, r, v0, theta, kappa, sigma, n_simulations=10000, n_steps=252):
#     dt = T / n_steps
#     call_prices = np.zeros(n_simulations)
# 
#     for i in range(n_simulations):
#         St = S
#         vt = np.zeros(n_steps)
#         vt[0] = v0
# 
#         for j in range(1, n_steps):
#             Z1 = np.random.normal()
#             Z2 = np.random.normal()
# 
#             dW1 = np.sqrt(dt) * Z1
#             dW2 = np.sqrt(dt) * Z2
# 
#             dSt_bs = r * St * dt + np.sqrt(vt[j-1]) * St * dW1
#             dvt_heston = kappa * (theta - vt[j-1]) * dt + sigma * np.sqrt(vt[j-1]) * dW2
# 
#             St += dSt_bs
#             vt[j] = vt[j-1] + dvt_heston
# 
#         call_price = max(St - K, 0) * np.exp(-r * T)
#         call_prices[i] = call_price
# 
#     return np.mean(call_prices)
# 
# 
# 
# 
# 
# 
# Black-Scholes:
# 
# La fórmula para calcular el precio de una opción de compra (C) utilizando el modelo de Black- Scholes es:
# 
# 
# 
# 
# Donde:
# 
# C = S ⋅ N (d1) − K ⋅ e−rT
# 
# ⋅ N (d2)
# 
# 
# S es el precio actual del activo subyacente.
# K es el precio de ejercicio de la opción.
# T es el tiempo hasta la expiración de la opción.
# r es la tasa de interés libre de riesgo.
# N es la función de distribución acumulada de la distribución normal estándar.
# d1 y d2 se calculan como sigue:
# 
# 
# 
# 
# 
# 
# Heston:
# 
# d1 =
# 
# ln(S/K) + (r + 1 σ2)T
# σ d2 = d1 − σ
# 
# La fórmula para calcular el precio de una opción de compra con volatilidad no constante utilizando el modelo de Heston es más compleja y generalmente se calcula mediante métodos numéricos debido a su naturaleza estocástica. La dinámica del modelo sigue un sistema de ecuaciones
# diferenciales estocásticas (SDES), las cuales describen la evolución del precio del activo y de la volatilidad en el tiempo.
# El modelo de Heston utiliza las siguientes ecuaciones diferenciales estocásticas (SDES):
# 
# 
# dSt = rStdt +
# 
# vt StdW 1
# 
# dvt = κ(θ − vt)dt + σ
# 
# vt dW 2
# 
# 
# Donde:
# St es el precio del activo subyacente en el tiempo t.
# vt es la varianza estocástica (volatilidad al cuadrado) en el tiempo t. r es la tasa de interés libre de riesgo.
# κ es la velocidad de reversión hacia la media. θ es la media a largo plazo de la volatilidad. σ es la volatilidad de la volatilidad.
# W 1 y W 2 son procesos de Wiener (movimiento Browniano).
#         t	       t
# 
# 
# Modelo Híbrido Tradicional Black-Scholes-Heston:
# El modelo híbrido combina los aspectos deterministas del modelo de Black-Scholes con la
# volatilidad estocástica del modelo de Heston. La dinámica de precios del modelo híbrido se expresa mediante el siguiente sistema de ecuaciones diferenciales estocásticas (SDES):
# 
# 
# dSt = (rStdt +
# 
# vt StdW 1)BS
# 
# dvt = (κ(θ − vt)dt + σ
# 
# vt dW 2)Heston
# 
# 
# Donde los subíndices BS y Heston indican las contribuciones de los modelos de Black-Scholes y
# Heston, respectivamente. El precio de la opción se calcula utilizando el precio del activo subyacente
# obtenido del modelo de Black-Scholes y la volatilidad estocástica del modelo de Heston.
# 
# 
# Descripcion detallada de la diferencia entre modelo de Heston e Hibrido. Un análisis matemático.
# 
# En resumen, la diferencia principal entre el modelo de Heston y el modelo híbrido radica en cómo manejan la volatilidad.
# 
# 
# •	Modelo de Heston: Utiliza un enfoque completamente estocástico para modelar la volatilidad. La volatilidad en este modelo sigue una dinámica propia, descrita por una ecuación diferencial estocástica (SDE). Es un modelo más flexible que puede capturar mejor la sonrisa de volatilidad y otros fenómenos observados en los mercados financieros.
# •	Modelo Híbrido: Combina la simplicidad del modelo de Black-Scholes para la dinámica del precio del activo subyacente con la complejidad del modelo de Heston para la volatilidad. Esto permite capturar tanto la dinámica determinista como estocástica en el precio de las opciones. Es una manera de aprovechar las fortalezas de ambos modelos para obtener una mejor representación de los precios de las opciones en el mercado real.
# 
# En términos matemáticos, el modelo de Heston utiliza ecuaciones diferenciales estocásticas (SDES) para describir la evolución del precio del activo y la volatilidad, mientras que el modelo híbrido combina las SDES del modelo de Heston con la dinámica determinista del modelo de Black-Scholes. Esto se refleja en las ecuaciones que describen la dinámica del activo y la volatilidad en cada modelo.
# 

# ![image.png](data:image/png;base64,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)

# Expandamos las ecuaciones del modelo de Heston y del modelo híbrido:
# 
# Modelo de Heston:
# 
# 
# dSt = rStdt +
# 
# vt StdW 1
# 
# dvt = κ(θ − vt)dt + σ
# 
# vt dW 2
# 
# 
# 
# Expandiendo cada término, tenemos:
# 
# Para dSt:
# 
#  dSt = rStdt +
# 
# vt StdW 1
# 
# 
# Para dvt:
# 
# dvt = κ(θ − vt)dt + σ
# 
# vt dW 2
# 
# 
# Modelo Híbrido:
# 
# dSt = (rStdt +
# 
# vt StdW 1)BS
# 
# dvt = (κ(θ − vt)dt + σ
# 
# vt dW 2)Heston
# 
# 
# Expresando cada término dentro de los paréntesis, tenemos:
# 
# Para dSt:
# 
# (rStdt +
# 
# vt StdW 1)BS = rStdt +
# 
# vt StdW 1
# 
# Para dvt:
# 
# (κ(θ − vt)dt + σ
# 
# vt dW 2)Heston = κ(θ − vt)dt + σ
# 
# vt dW 2
# 
# En el modelo híbrido, los términos dentro de los paréntesis indican la procedencia de los incrementos estocásticos dW 1 y dW 2, donde dW 1 proviene del modelo de Black-Scholes y dW 2
# t	t	t	t
# proviene del modelo de Heston.
# 
# A la pregunta de qué entonces mediante la formula el modelo de Heston seria exactamente igual al modelo hibrido, la respuesta seria negativa, los modelos de Heston y híbrido no son exactamente iguales. La diferencia clave radica en cómo se modela la evolución de la volatilidad (vt).
# 
# En el modelo de Heston, la volatilidad sigue una dinámica estocástica determinada por la ecuación:
# 
#  Mientras que, en el modelo híbrido, se mantiene la misma ecuación para la volatilidad, pero el precio del activo (St) sigue la dinámica del modelo de Black-Scholes, lo que significa que el término (dSt)
#   se calcula de acuerdo con la fórmula de Black-Scholes y no se ve afectado por la volatilidad estocástica, como en el modelo de Heston.
# 
# Por lo tanto, aunque comparten la misma ecuación para la evolución de la volatilidad, la forma en que se modela el precio del activo subyacente es diferente entre el modelo de Heston y el modelo híbrido.
# 
#  Entonces al demostrar la diferencia exacta entre ambos modelos, la diferencia clave entre el modelo de Heston y el modelo híbrido radica en cómo se modela la evolución de la volatilidad (vt). En el modelo de Heston, la volatilidad sigue una dinámica
# estocástica determinada por la ecuación:
# 
# 
# dvt = κ(θ − vt)dt + σ
# 
# Donde:
# 
# vt dW Heston
# 
# vt es la volatilidad estocástica en el tiempo t. κ es la velocidad de reversión hacia la media. θ es la media a largo plazo de la volatilidad. σ es la volatilidad de la volatilidad.
# dW Heston es un proceso de Wiener (movimiento Browniano).
# 
# En contraste, en el modelo híbrido, la evolución de la volatilidad se mantiene igual a la del modelo de Heston, pero la evolución del precio del activo subyacente se ajusta para reflejar el modelo de Black-Scholes. En términos matemáticos, esto se expresa como:
# 
# 
# dSt = (rStdt +
# 
# Donde:
# 
#   vt StdW1t)BS
# 
# St es el precio del activo subyacente en el tiempo t. r es la tasa de interés libre de riesgo.
# vt es la volatilidad estocástica en el tiempo t.
# W1t es un proceso de Wiener (movimiento Browniano).
# 
# La diferencia radica en cómo se integra la volatilidad estocástica en la dinámica del precio del activo subyacente: en el modelo de Heston, se utiliza la dinámica completa de Heston, mientras
# que, en el modelo híbrido, se combina la dinámica del precio del activo subyacente del modelo de Black-Scholes con la volatilidad estocástica del modelo de Heston.
# 
# A la pregunta de qué dado que la diferencia radica en cómo se integra la volatilidad estocástica en la dinámica del precio del activo subyacente: en el modelo de Heston, se utiliza la dinámica completa de Heston, mientras que, en el modelo híbrido, se combina la dinámica del precio del activo subyacente del modelo de Black-Scholes con la volatilidad estocástica del modelo de Heston. El modelo hibrido sería más eficiente en medir con precisión tanto la dinámica del precio del activo subyacente, así como la volatilidad estocástica. Tenemos que la respuesta en teoría, el modelo híbrido puede ser más eficiente para capturar tanto la dinámica del precio del activo subyacente como la volatilidad estocástica en ciertos contextos. La razón principal es que el modelo híbrido combina lo mejor de ambos mundos: utiliza la simplicidad y eficiencia computacional del modelo de Black-Scholes para modelar la dinámica del precio del activo subyacente, mientras que incorpora la complejidad y la capacidad de captura de la volatilidad estocástica del modelo de Heston.
# 
# Sin embargo, la eficiencia del modelo híbrido dependerá en gran medida de la precisión de los parámetros utilizados y de la calidad de los datos de entrada. En algunos casos, el modelo de Heston puede ser más adecuado si la volatilidad estocástica es el factor dominante en la dinámica del precio del activo subyacente, mientras que, en otros casos, el modelo de Black-Scholes podría ser suficiente si la volatilidad se considera relativamente constante o si se desea una simulación más rápida y menos intensiva en recursos.
# 
# En resumen, la eficiencia y precisión del modelo híbrido dependen de la naturaleza específica del activo subyacente, las condiciones del mercado y los objetivos del análisis financiero. Es importante evaluar cuidadosamente cada modelo en función de las necesidades y características del problema en cuestión.
# 
# Variables fundamentales:
# 
# En el modelo de Heston, se utiliza la dinámica completa de Heston, mientras que, en el modelo híbrido, se combina la dinámica del precio del activo subyacente del modelo de Black-Scholes con la volatilidad estocástica del modelo de Heston. En el modelo de Black-Scholes, la variable que describe la dinámica del precio del activo subyacente es St, que representa el precio del activo en el tiempo t.
# En el modelo de Heston, la variable que describe la volatilidad estocástica es vt, que representa la varianza estocástica (volatilidad al cuadrado) en el tiempo t.
# 
# Es decir:
# 
# Del Modelo hibrido, St: precio del activo en el tiempo t.
#          Del modelo Black-Scholes, Vt: la varianza estocástica (volatilidad al cuadrado) en el tiempo t.
#          Del modelo Heston, la dinámica completa.
# 
# **Modelo Híbrido LSTM-Heston con componentes de redes neuronales:**
# 
# Vamos a profundizar en la explicación matemática de cada componente y cómo se integran en el modelo híbrido.
# Modelo de Heston: La dinámica del modelo de Heston está dada por las siguientes ecuaciones estocásticas diferenciales (SDEs):
# 
# dSt=rStdt+vtStdW
# 
# dvt=κ(θ−vt)dt+σvtdW2
# 
# donde:
# •	St es el precio del activo subyacente en el tiempo t.
# •	vt es la volatilidad estocástica (varianza) en el tiempo t.
# •	r es la tasa de interés libre de riesgo.
# •	κ es la velocidad de reversión hacia la media.
# •	θ es la media a largo plazo de la volatilidad.
# •	σ es la volatilidad de la volatilidad.
# •	dW1 y dW2 son incrementos infinitesimales de procesos de Wiener (movimiento Browniano).
# 
# Proceso de Simulación de Heston: La volatilidad vt se simula iterativamente para cada paso de tiempo dt utilizando las ecuaciones de Heston. Este proceso proporciona una serie de volatilidades que capturan la dinámica estocástica de la volatilidad a lo largo del tiempo.
# 
# Modelo LSTM: La red LSTM es una arquitectura de red neuronal que procesa secuencias temporales de datos. En este caso, se utiliza para aprender patrones en la secuencia de volatilidades generadas por el modelo de Heston.
# 
# Proceso de Preparación para la LSTM: Se crea una secuencia de entrada para la LSTM (Xlstm) utilizando las últimas 10 volatilidades generadas por el modelo de Heston. Esto permite a la LSTM aprender de la historia reciente de la volatilidad.
# Predicción de la LSTM: La secuencia Xlstm se pasa
#  a través del modelo LSTM, y se obtiene una predicción (lstm_output) que representa la proyección de la volatilidad futura según el aprendizaje de patrones de la LSTM.
# 
# Cálculo del Precio de la Opción: El precio de la opción en cada simulación se calcula mediante la fórmula de opción de compra Black-Scholes modificada, donde la volatilidad (σ) se ajusta utilizando la proyección de la LSTM (lstm_output). El precio de la opción se agrega al vector de precios (call_prices).
# 
# Resultado Final: La función devuelve el precio promedio de la opción de compra sobre todas las simulaciones.
# 
# Ejemplo de Uso: Se proporciona un ejemplo de cómo llamar a la función, especificando los parámetros necesarios, incluyendo el modelo LSTM y los parámetros del modelo de Heston.
# 
# **Comentarios Adicionales: **
# 
# •	La integración de la LSTM permite al modelo adaptarse a patrones no lineales y complejidades en la secuencia de volatilidades generadas por el modelo de Heston.
# 
# •	La fórmula de Black-Scholes se modifica para incorporar la volatilidad proyectada por la LSTM, buscando mejorar la precisión de la valoración de opciones al capturar tanto la dinámica del precio como la complejidad de la volatilidad estocástica.
# 
# **Diferencia entre los modelos híbridos tradicional y de aprendizaje profundo.**
# 
# La principal diferencia entre el Modelo Híbrido Tradicional Black-Scholes-Heston y el Modelo Híbrido LSTM-Heston con componentes de redes neuronales radica en la forma en que se modela y procesa la información de volatilidad.
# 
# En el Modelo Híbrido Tradicional Black-Scholes-Heston, la dinámica de la volatilidad se sigue utilizando el modelo estocástico de Heston, que es conocido por capturar fenómenos observados en los mercados financieros, como la sonrisa de volatilidad. La parte determinista del modelo se basa en la fórmula de Black-Scholes para la dinámica del precio del activo subyacente.
# 
# En cambio, en el Modelo Híbrido LSTM-Heston con componentes de redes neuronales, se incorpora una red neuronal LSTM para aprender y modelar patrones no lineales en la secuencia de volatilidades generadas por el modelo de Heston. Esto permite al modelo adaptarse a complejidades no lineales y aprender relaciones temporales en los datos de volatilidad.
# 
# En resumen, la diferencia clave es la introducción de la red neuronal LSTM en el segundo modelo, lo que le brinda al modelo la capacidad de aprender patrones más complejos en la secuencia temporal de volatilidades, potencialmente mejorando la capacidad de predicción y adaptación a condiciones no lineales en comparación con el enfoque más tradicional del primer modelo.
# 
# **Matemáticamente: **
# 
# Matemáticamente, la diferencia entre el Modelo Híbrido Tradicional Black-Scholes-Heston y el Modelo Híbrido LSTM-Heston con componentes de redes neuronales radica en cómo se modela la volatilidad y se incorpora la red neuronal LSTM en el segundo modelo. Aquí está la explicación matemática de cada modelo:
# 
# **Modelo Híbrido Tradicional Black-Scholes-Heston:**
# 
# La dinámica del precio del activo subyacente (St) y la volatilidad (vt) en el modelo híbrido tradicional sigue las ecuaciones diferenciales estocásticas (SDE) de Black-Scholes y Heston, respectivamente:
# Para el precio del activo (St):
# 
# dSt=rStdt+vtStdW
# 
# Para la volatilidad (vt):
# 
# dvt=κ(θ−vt)dt+σvtdW
# 
# Donde:
# •	r es la tasa de interés libre de riesgo.
# •	κ es la velocidad de reversión hacia la media.
# •	θ es la media a largo plazo de la volatilidad.
# •	σ es la volatilidad de la volatilidad.
# •	dW1 y dW2 son incrementos infinitesimales de procesos de Wiener (movimiento Browniano).
# 
# **Modelo Híbrido LSTM-Heston con componentes de redes neuronales:**
# 
# En este modelo, se introduce una red neuronal LSTM para procesar la secuencia de volatilidades generadas por el modelo de Heston. La red LSTM toma como entrada las últimas 10 volatilidades (vt) y aprende patrones en esa secuencia. La salida de la LSTM (lstm_output) se utiliza en la fórmula de Black-Scholes modificada para calcular el precio de la opción de compra.
# La fórmula de Black-Scholes modificada en este contexto sería:
# 
# Call_Price=max(St−K,0)e−rT+lstm_output
# 
# Donde:
# 
# •	K es el precio de ejercicio de la opción.
# •	T es el tiempo hasta la expiración de la opción.
# 
# La red neuronal LSTM agrega un componente adicional al modelo, permitiendo que el sistema aprenda patrones no lineales y complejidades en la secuencia temporal de volatilidades, mejorando así la capacidad del modelo para adaptarse a la información histórica y proporcionar predicciones más precisas.
# 
# 
# 
# ---
# 
# 

# In[ ]:





# In[ ]:





# In[23]:





# In[23]:




