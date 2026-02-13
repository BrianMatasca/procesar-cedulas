import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import time
from io import BytesIO
from typing import Dict, List, Tuple
import json
from collections import defaultdict

from settings import (
    LOGIN_URL,
    GET_USER_URL,
    CREATE_USER_URL,
    FIXED_VALUES,
    REQUIRED_FIELDS,
    CONNECT_TIMEOUT,
    REQUEST_TIMEOUT,
)


# Configuración de la página
st.set_page_config(
    page_title="Cargar Cédulas",
    page_icon="📋",
    layout="wide"
)


async def login_async(session: aiohttp.ClientSession, username: str, password: str) -> Tuple[bool, str, str]:
    """Realiza el login y retorna el token de forma asíncrona"""
    try:
        credentials = {
            "username": username,
            "password": password
        }
        
        async with session.post(LOGIN_URL, json=credentials) as response:
            if response.status != 200:
                return False, None, f"HTTP {response.status}"

            data = await response.json(content_type=None)

            if data.get("status") == 200:
                token = data.get("payload", {}).get("token")
                if token:
                    return True, token, "Login exitoso"

            return False, None, f"Respuesta inválida: {data}"

    except asyncio.TimeoutError:
        return False, None, "Timeout en login"
    except Exception as e:
        return False, None, f"Error en login: {str(e)}"


async def get_user_by_cedula_async(
    cedula: str,
    token: str,
    session: aiohttp.ClientSession
) -> Tuple[bool, Dict, str]:
    """Obtiene los datos del usuario por cédula de forma asíncrona"""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        async with session.get(f"{GET_USER_URL}{cedula}", headers=headers) as response:
            data = await response.json(content_type=None)

            api_status = data.get("status")
            payload = data.get("payload", {})

            if response.status == 200 and api_status == 200:
                return True, payload, "OK"

            if api_status == 400 and payload.get("message") == "El número de cédula ya se encuentra registrado.":
                return False, payload, "CEDULA_YA_REGISTRADA"

            if api_status != 200:
                return False, payload, payload.get("message", "Error de negocio")

            return False, {}, f"HTTP {response.status}"

    except asyncio.TimeoutError:
        return False, {}, "Timeout consultando cédula"
    except Exception as e:
        return False, {}, f"Error conexión: {str(e)}"


async def create_user_async(
    user_data: Dict,
    token: str,
    session: aiohttp.ClientSession
) -> Tuple[bool, str]:
    """Crea un usuario con los datos obtenidos de forma asíncrona"""
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        body = {
            "identity_card": user_data.get("identity_card"),
            "email": None,
            "phone": None,
            "opciones_respuesta_id": FIXED_VALUES["opciones_respuesta_id"],
            "company_nit": None,
            "first_name": user_data.get("first_name"),
            "last_name": user_data.get("last_name"),
            "second_last_name": user_data.get("second_last_name"),
            "department_description": user_data.get("department_description"),
            "municipality_description": user_data.get("municipality_description"),
            "polling_station": user_data.get("polling_station"),
            "commune_description": user_data.get("commune_description"),
            "neighborhood_description": user_data.get("neighborhood_description"),
            "polling_table": user_data.get("polling_table"),
            "address_description": user_data.get("address_description"),
            "rol_description": FIXED_VALUES["rol_description"],
            "leader_id": FIXED_VALUES["leader_id"],
            "optional_data": [],
            "lugarde_recidencia": None
        }

        async with session.post(CREATE_USER_URL, json=body, headers=headers) as response:
            if response.status in (200, 201):
                return True, "Creado exitosamente"
            if response.status == 409:
                return False, "Usuario ya existe"

            raw = await response.read()
            return False, f"HTTP {response.status}: {raw[:200]}"

    except asyncio.TimeoutError:
        return False, "Timeout creando usuario"
    except Exception as e:
        return False, f"Error creación: {str(e)}"


def normalize_cedula(raw):
    """Normaliza la cédula a formato string numérico"""
    try:
        return str(int(float(raw)))
    except:
        return None


def validate_user_data(user_data: Dict) -> Tuple[bool, str]:
    pending_fields = [
        field for field in REQUIRED_FIELDS
        if str(user_data.get(field)).strip().upper() == "PENDIENTE"
    ]

    if pending_fields:
        return False, f"Datos incompletos (PENDIENTE): {', '.join(pending_fields)}"

    return True, ""


def categorize_error(error_msg: str) -> str:
    """Categoriza el error para mejor seguimiento"""
    if "CEDULA_YA_REGISTRADA" in error_msg:
        return "Cédula ya registrada"
    elif "PENDIENTE" in error_msg:
        return "Datos PENDIENTE"
    elif "Timeout" in error_msg:
        return "Timeout"
    elif "Usuario ya existe" in error_msg:
        return "Usuario duplicado"
    elif "Cédula inválida" in error_msg:
        return "Formato inválido"
    elif "Error conexión" in error_msg:
        return "Error de conexión"
    else:
        return "Otros errores"


async def process_cedula(
    cedula: str, 
    token: str, 
    session: aiohttp.ClientSession
) -> Tuple[bool, Dict]:
    """
    Procesa UNA cédula de forma secuencial
    """
    
    cedula_clean = normalize_cedula(cedula)
    if not cedula_clean:
        return False, {
            'cedula': str(cedula),
            'razon': 'Cédula inválida (formato)'
        }
    
    # 1) Obtener datos del usuario
    success, user_data, error_msg = await get_user_by_cedula_async(
        cedula_clean, token, session
    )
    
    if not success:
        return False, {
            'cedula': cedula_clean,
            'razon': f"Consulta fallida: {error_msg}"
        }
    
    # 2) Validar campos PENDIENTE
    is_valid, validation_error = validate_user_data(user_data)
    if not is_valid:
        return False, {
            'cedula': cedula_clean,
            'razon': validation_error
        }
    
    # 3) Crear usuario
    success, message = await create_user_async(user_data, token, session)
    
    if success:
        return True, {
            'cedula': cedula_clean,
            'nombre': f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}",
            'municipio': user_data.get('municipality_description', '')
        }
    else:
        return False, {
            'cedula': cedula_clean,
            'razon': f"Creación fallida: {message}"
        }


def format_time(seconds: float) -> str:
    """Formatea segundos a formato legible"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


async def process_all_cedulas_sequential(
    cedulas: List[str],
    token: str,
    progress_callback
) -> Tuple[List[Dict], List[Dict]]:
    """
    Procesamiento secuencial optimizado con seguimiento de errores en tiempo real
    """
    successful = []
    failed = []
    
    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=CONNECT_TIMEOUT,
        sock_read=REQUEST_TIMEOUT
    )
    
    connector = aiohttp.TCPConnector(
        limit=1,
        limit_per_host=1,
        ttl_dns_cache=300,
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        
        for idx, cedula in enumerate(cedulas):
            
            success, result = await process_cedula(cedula, token, session)
            
            if success:
                successful.append(result)
            else:
                failed.append(result)
            
            # Actualizar progreso con información de errores
            if progress_callback:
                progress_callback(idx + 1, len(cedulas), successful, failed)
    
    return successful, failed


def process_excel_sequential(df: pd.DataFrame, token: str) -> Tuple[List[Dict], List[Dict], float]:
    """UN SOLO asyncio.run() para TODO el proceso secuencial"""
    
    total = len(df)
    cedulas = df['cedula'].tolist()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_text = st.empty()
    
    # Contenedor para errores en tiempo real
    error_container = st.container()
    with error_container:
        error_header = st.empty()
        error_metrics_container = st.empty()
    
    start_time = time.time()
    
    def update_progress(completed, total_count, successful, failed):
        elapsed_time = time.time() - start_time
        progress = completed / total_count
        progress_bar.progress(min(progress, 1.0))
        
        rate = completed / elapsed_time if elapsed_time > 0 else 0
        eta = (total_count - completed) / rate if rate > 0 else 0
        
        status_text.text(f"Procesados: {completed}/{total_count} | Velocidad: {rate * 60:.2f} cédulas/min")
        stats_text.markdown(
            f"**Exitosos:** {len(successful)} | **Fallidos:** {len(failed)} | "
            f"**Tiempo transcurrido:** {format_time(elapsed_time)} | "
            f"**ETA:** {format_time(eta)}"
        )
        
        # Mostrar resumen de errores en tiempo real
        if failed:
            error_header.markdown("### 📊 Errores en Tiempo Real")
            
            # Categorizar errores
            error_counts = defaultdict(int)
            for error_record in failed:
                category = categorize_error(error_record['razon'])
                error_counts[category] += 1
            
            # Mostrar métricas de errores
            num_categories = len(error_counts)
            cols = error_metrics_container.columns(min(num_categories, 4))
            
            for idx, (category, count) in enumerate(sorted(error_counts.items(), key=lambda x: -x[1])):
                col_idx = idx % 4
                with cols[col_idx]:
                    st.metric(category, count, delta=None)
    
    successful, failed = asyncio.run(
        process_all_cedulas_sequential(cedulas, token, update_progress)
    )
    
    total_time = time.time() - start_time
    
    progress_bar.progress(1.0)
    status_text.text(f"¡Procesamiento completado! Procesados: {total}/{total}")
    stats_text.markdown(
        f"**Total exitosos:** {len(successful)} | **Total fallidos:** {len(failed)} | "
        f"**Tiempo total:** {format_time(total_time)} | "
        f"**Velocidad promedio:** {total/total_time:.2f} cédulas/seg"
    )
    
    return successful, failed, total_time


def login(username: str, password: str) -> Tuple[bool, str, str]:
    """Realiza el login síncrono para la UI"""
    async def _login():
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await login_async(session, username, password)
    
    try:
        return asyncio.run(_login())
    except Exception as e:
        return False, None, f"Error al hacer login: {str(e)}"


def create_failed_excel(failed_records: List[Dict]) -> BytesIO:
    """Crea un Excel con los registros fallidos"""
    df_failed = pd.DataFrame(failed_records)
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_failed.to_excel(writer, index=False, sheet_name='Registros Fallidos')
    
    output.seek(0)
    return output


def main():
    st.title("📋 Procesador de Cédulas")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración")
    st.sidebar.metric("Modo", "Secuencial (1 a la vez)")
    st.sidebar.metric("Timeout de lectura", f"{REQUEST_TIMEOUT}s")
    st.sidebar.metric("Timeout de conexión", f"{CONNECT_TIMEOUT}s")
    
    st.sidebar.info(
        "**Procesamiento secuencial optimizado:**\n\n"
        "• 1 cédula a la vez (sin concurrencia)\n"
        "• Sesión HTTP compartida (keep-alive)\n"
        "• Connection pooling activo\n"
        "• Seguimiento de errores en tiempo real"
    )
    
    # Credenciales de login
    st.subheader("🔐 Credenciales de Acceso")
    
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input(
            "Usuario / Email",
            placeholder="ejemplo@correo.com",
            help="Ingrese su nombre de usuario o email"
        )
    
    with col2:
        password = st.text_input(
            "Contraseña",
            type="password",
            placeholder="••••••••",
            help="Ingrese su contraseña"
        )
    
    # Validar que se ingresaron credenciales
    credentials_provided = bool(username and password)
    
    if not credentials_provided:
        st.warning("⚠️ Por favor ingrese sus credenciales para continuar")
    
    st.markdown("---")
    
    # Upload file
    uploaded_file = st.file_uploader(
        "Cargar archivo Excel con columna 'cedula'",
        type=['xlsx', 'xls'],
        help="El archivo debe contener una columna llamada 'cedula'",
        disabled=not credentials_provided
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            if 'cedula' not in df.columns:
                st.error("❌ El archivo debe contener una columna llamada 'cedula'")
                return
            
            st.success(f"✅ Archivo cargado correctamente: {len(df):,} registros encontrados")
            
            with st.expander("👀 Vista previa del archivo"):
                st.dataframe(df.head(10))
            
            # Botón para procesar
            if st.button("Procesar Cédulas", type="primary", use_container_width=True):
                
                # Validar credenciales con login
                with st.spinner("Verificando credenciales..."):
                    success, token, message = login(username, password)
                
                if not success:
                    st.error(f"❌ No se pudo autenticar")
                    st.error(f"Detalle: {message}")
                    st.warning("Verifique que su usuario y contraseña sean correctos")
                    return
                
                st.success(f"✅ {message}")
                
                # Procesar
                st.markdown("### 🔄 Procesamiento en curso (secuencial)...")
                successful, failed, total_time = process_excel_sequential(df, token)
                
                # Mostrar resultados finales
                st.markdown("---")
                st.markdown("## 📊 Resultados Finales del Procesamiento")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Procesados", f"{len(df):,}")
                
                with col2:
                    st.metric("✅ Exitosos", f"{len(successful):,}")
                
                with col3:
                    st.metric("❌ Fallidos", f"{len(failed):,}")
                
                with col4:
                    st.metric("⏱️ Tiempo Total", format_time(total_time))
                
                col5, col6 = st.columns(2)
                with col5:
                    success_rate = (len(successful) / len(df) * 100) if len(df) > 0 else 0
                    st.metric("Tasa de éxito", f"{success_rate:.1f}%")
                
                with col6:
                    processing_speed = len(df) / total_time if total_time > 0 else 0
                    st.metric("Velocidad promedio", f"{processing_speed*60:.2f} cédulas/min")
                
                # Análisis detallado de errores
                if failed:
                    st.markdown("### 🔍 Análisis Detallado de Errores")
                    
                    df_failed = pd.DataFrame(failed)
                    error_counts = df_failed['razon'].value_counts()
                    
                    col_err1, col_err2 = st.columns(2)
                    
                    with col_err1:
                        st.markdown("**Top 10 Tipos de Error:**")
                        for error, count in error_counts.head(10).items():
                            st.text(f"• {error}: {count}")
                    
                    with col_err2:
                        st.markdown("**Distribución:**")
                        st.bar_chart(error_counts.head(10))
                                        
                
                # Mostrar exitosos
                if successful:
                    with st.expander(f"✅ Ver registros exitosos ({len(successful):,})"):
                        st.dataframe(pd.DataFrame(successful))
                
                # Descargar fallidos
                if failed:
                    with st.expander(f"❌ Ver registros fallidos ({len(failed):,})"):
                        st.dataframe(pd.DataFrame(failed))
                    
                    failed_excel = create_failed_excel(failed)
                    
                    st.download_button(
                        label="📥 Descargar Excel con Registros Fallidos",
                        data=failed_excel,
                        file_name="registros_fallidos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                else:
                    st.success("🎉 ¡Todos los registros fueron procesados exitosamente!")
                
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()