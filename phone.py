from dataclasses import dataclass

@dataclass
class Phone:
    id: int
    brand: str
    model: str
    price: int
    storage: float
    RAM: int
    battery_capacity: int
    weight: int
    rating: float
    description: str

    def set_task(self):
        return {
            "id": self.id,
            "brand": self.brand,
            "model": self.model,
            "price": self.price,
            "storage": self.storage,
            "RAM": self.RAM,
            "battery_capacity": self.battery_capacity,
            "weight": self.weight,
            "rating": self.rating,
            "description": self.description,
        }
    
    @staticmethod
    def get_phone(data):
        return Phone(**data)