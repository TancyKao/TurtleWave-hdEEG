import os
import pandas as pd
from datetime import datetime

class ResultManager:
    """Class to manage review results"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results = {}  # {event_id: {'decision': 'accept'|'reject', 'comment': '...', 'timestamp': ...}}
    
    def add_result(self, event_id, decision, comment=""):
        """Add a review result"""
        self.results[event_id] = {
            'decision': decision,
            'comment': comment,
            'timestamp': datetime.now()
        }
    
    def get_result(self, event_id):
        """Get the review result for an event"""
        return self.results.get(event_id)
    
    def get_results_count(self):
        """Get count of reviewed events"""
        return len(self.results)
    
    def get_accepted_count(self):
        """Get count of accepted events"""
        return sum(1 for r in self.results.values() if r['decision'] == 'accept')
    
    def get_rejected_count(self):
        """Get count of rejected events"""
        return sum(1 for r in self.results.values() if r['decision'] == 'reject')
    
    def save_results(self, event_type, events_df):
        """Save review results to CSV"""
        if not self.results:
            return False, "No results to save"
            
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(self.output_dir, f"{event_type}_review")
            os.makedirs(results_dir, exist_ok=True)
            
            # Create results dataframe
            results_list = []
            for event_id, result in self.results.items():
                if event_id in events_df.index:
                    event = events_df.loc[event_id]
                    results_list.append({
                        'event_id': event_id,
                        'start_time': event['start_time'],
                        'end_time': event['end_time'],
                        'channel': event['channel'],
                        'confidence': event.get('confidence', 1.0),
                        'decision': result['decision'],
                        'comment': result.get('comment', ''),
                        'review_time': result.get('timestamp', datetime.now())
                    })
            
            if not results_list:
                return False, "No valid results to save"
                
            results_df = pd.DataFrame(results_list)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{event_type}_review_results_{timestamp}.csv"
            filepath = os.path.join(results_dir, filename)
            
            results_df.to_csv(filepath, index=False)
            return True, filepath
            
        except Exception as e:
            return False, f"Error saving results: {str(e)}"
    
    def load_results(self, filepath):
        """Load previously saved results"""
        try:
            results_df = pd.read_csv(filepath)
            
            # Convert to our internal format
            self.results = {}
            for _, row in results_df.iterrows():
                self.results[row['event_id']] = {
                    'decision': row['decision'],
                    'comment': row.get('comment', ''),
                    'timestamp': row.get('review_time', datetime.now())
                }
            
            return True, f"Loaded {len(self.results)} results"
            
        except Exception as e:
            return False, f"Error loading results: {str(e)}"