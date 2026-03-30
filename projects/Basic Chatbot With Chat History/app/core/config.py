from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GROQ_API_KEY: str

    config_model = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )


settings = Settings()
