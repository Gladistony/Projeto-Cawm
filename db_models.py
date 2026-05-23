"""
Modelos SQLAlchemy para o CAWM.

Define a estrutura das tabelas usando ORM do SQLAlchemy.
"""
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Bacia(Base):
    """Modelo para representar uma bacia hidrográfica."""

    __tablename__ = "bacias"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nome = Column(String(255), unique=True, nullable=False, index=True)
    regiao = Column(String(255), nullable=True)
    area_km2 = Column(Float, nullable=True)

    # Parâmetros hidrológicos
    k = Column(Float, nullable=True)
    a = Column(Float, nullable=True)
    expo_perdas = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    kg = Column(Float, nullable=True)
    rio = Column(Integer, nullable=True)
    submax = Column(Float, nullable=True)
    gmax = Column(Float, nullable=True)

    # Condições iniciais (estado inicial dos reservatórios)
    reserva_solo_inicial = Column(Float, nullable=True)
    profundo_inicial = Column(Float, nullable=True)
    s3_inicial = Column(Float, nullable=True)
    s1_inicial = Column(Float, default=0.0)
    s2_inicial = Column(Float, default=0.0)

    # Constantes fixas
    b = Column(Float, default=1.666666667)
    T = Column(Float, default=86400.0)

    def __repr__(self):
        return f"<Bacia(id={self.id}, nome='{self.nome}', area_km2={self.area_km2})>"

    def to_dict(self):
        """Converte a bacia em dicionário para uso em funcoes.py."""
        return {
            "id": self.id,
            "nome": self.nome,
            "regiao": self.regiao,
            "area_km2": self.area_km2,
            "k": self.k,
            "a": self.a,
            "expo_perdas": self.expo_perdas,
            "beta": self.beta,
            "kg": self.kg,
            "rio": self.rio,
            "submax": self.submax,
            "gmax": self.gmax,
            "reserva_solo_inicial": self.reserva_solo_inicial,
            "profundo_inicial": self.profundo_inicial,
            "s3_inicial": self.s3_inicial,
            "s1_inicial": self.s1_inicial,
            "s2_inicial": self.s2_inicial,
            "b": self.b,
            "T": self.T,
        }


class CalibrationPeriod(Base):
    """Períodos de calibração/validação vinculados a uma bacia."""

    __tablename__ = "calibration_periods"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bacia_id = Column(Integer, ForeignKey("bacias.id"), nullable=False, index=True)
    station = Column(String(255), nullable=True)
    method = Column(String(32), nullable=True)

    calib_start = Column(Date, nullable=True)
    calib_end = Column(Date, nullable=True)
    val_start = Column(Date, nullable=True)
    val_end = Column(Date, nullable=True)

    def __repr__(self):
        return (
            f"<CalibrationPeriod(id={self.id}, bacia_id={self.bacia_id}, station='{self.station}',"
            f" method='{self.method}')>"
        )


class ModelResult(Base):
    """Resultados de desempenho do modelo para um período de calibração/validação."""

    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    calibration_period_id = Column(Integer, ForeignKey("calibration_periods.id"), nullable=False, index=True, unique=True)
    
    # Parâmetros do modelo
    area_km2 = Column(Float, nullable=True)
    s_mm = Column(Float, nullable=True)
    ks = Column(Float, nullable=True)
    
    # Métricas de desempenho - Calibração
    nse_calib = Column(Float, nullable=True)
    nse_sqrt_calib = Column(Float, nullable=True)
    nse_log_calib = Column(Float, nullable=True)
    pbias_calib = Column(Float, nullable=True)
    
    # Métricas de desempenho - Validação
    nse_val = Column(Float, nullable=True)
    nse_sqrt_val = Column(Float, nullable=True)
    nse_log_val = Column(Float, nullable=True)
    pbias_val = Column(Float, nullable=True)
    
    def __repr__(self):
        return (
            f"<ModelResult(id={self.id}, calib_period_id={self.calibration_period_id}, "
            f"nse_calib={self.nse_calib}, nse_val={self.nse_val})>"
        )


class EvaporationMonthly(Base):
    """Evaporação mensal por bacia.

    Mantém exatamente 12 registros por bacia, um para cada mês.
    """

    __tablename__ = "evaporation_monthly"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bacia_id = Column(Integer, ForeignKey("bacias.id"), nullable=False, index=True)
    mes = Column(Integer, nullable=False)
    valor = Column(Float, nullable=False)
    __table_args__ = (
        UniqueConstraint("bacia_id", "mes", name="uq_evaporation_monthly_bacia_mes"),
    )

    def __repr__(self):
        return f"<EvaporationMonthly(id={self.id}, bacia_id={self.bacia_id}, mes={self.mes}, valor={self.valor})>"


class PrecipitationDaily(Base):
    """Precipitação diária por bacia."""

    __tablename__ = "precipitation_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bacia_id = Column(Integer, ForeignKey("bacias.id"), nullable=False, index=True)
    data = Column(Date, nullable=False)
    valor = Column(Float, nullable=False)
    __table_args__ = (
        UniqueConstraint("bacia_id", "data", name="uq_precipitation_daily_bacia_data"),
    )

    def __repr__(self):
        return f"<PrecipitationDaily(id={self.id}, bacia_id={self.bacia_id}, data={self.data}, valor={self.valor})>"


class FlowDaily(Base):
    """Vazão observada diária (PAO) por bacia."""

    __tablename__ = "flow_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bacia_id = Column(Integer, ForeignKey("bacias.id"), nullable=False, index=True)
    data = Column(Date, nullable=False)
    valor = Column(Float, nullable=False)
    __table_args__ = (
        UniqueConstraint("bacia_id", "data", name="uq_flow_daily_bacia_data"),
    )

    def __repr__(self):
        return f"<FlowDaily(id={self.id}, bacia_id={self.bacia_id}, data={self.data}, valor={self.valor})>"
