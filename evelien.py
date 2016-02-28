import evelien_configurator
import evelien_analyzer

class factory:
    
    name = 'Evelien'
                        
    def create_configurator(self, root, frame):        
        return evelien_configurator.EvelienConfigurator(root, frame)        
        
    def create_analyzer(self):        
        return evelien_analyzer.EvelienAnalyzer()        
    