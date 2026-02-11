# URLs y credenciales
LOGIN_URL = "https://prod-api-directory.playtechla.com.co/api/directory/anasanclemente/users/login"
GET_USER_URL = "https://prod-api-directory.playtechla.com.co/api/directory/anasanclemente/users/get-by-identity-card/"
CREATE_USER_URL = "https://prod-api-directory.playtechla.com.co/api/directory/anasanclemente/users/create"

# Valores fijos para crear usuario
FIXED_VALUES = {
    # Conocido por medio de un lider
    "opciones_respuesta_id": 6,
    "leader_id": 4235,
    "rol_description": "SIMPATIZANTE"
}

# Campos que NO pueden ser "PENDIENTE" o vacíos
REQUIRED_FIELDS = [
    "first_name",
    "last_name",
    "second_last_name",
    "department_description",
    "municipality_description",
    "polling_station",
    "polling_table",
    "address_description"
    "rol_description"
]

CONNECT_TIMEOUT = 15  # Timeout de conexión (Resolver DNS, abrir socket TCP)
REQUEST_TIMEOUT = 45  # Timeout por petición (Esperar respuesta del servidor)

