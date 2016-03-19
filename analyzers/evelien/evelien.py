import evelien_configurator
import evelien_analyzer
import evelien_decorator

class factory:
    
    name = 'Evelien'
                        
    def create_configuration(self):
        return evelien_analyzer.EvelienAnalyzer.Configuration()        
        
    def create_configurator(self, config, host, root, frame):        
        return evelien_configurator.EvelienConfigurator(config, host, root, frame)        
        
    def create_analyzer(self, config, logger):        
        return evelien_analyzer.EvelienAnalyzer(config, logger)        

    def create_decorator(self, config):        
        return evelien_decorator.EvelienDecorator(config)        
    