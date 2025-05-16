class ServiceManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.services = {}
    
    def register(self, name, service):
        self.services[name] = service
    
    def get(self, name):
        return self.services.get(name)
