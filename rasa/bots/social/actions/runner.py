from threading import Thread
import asyncio
from abc import abstractmethod

class Runner(Thread):
    """
    Thread used to execute co-routines.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(daemon=True)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.start()
    
    def run(self):
        self.loop.run_forever()
    
    def _is_running_task(self):
        """
        Checks if at least a co-routine is running.
        
        Returns:
            True if at least a co-routine is running
            False else.
        """

        return len(asyncio.all_tasks(self.loop)) != 0
    
    @abstractmethod
    def add_task(self, *args, **kwargs) -> asyncio.Future:
        """
        This method must be implemented by child.
        It is used to create and run an async task to loop handled by this thread (this class is a thread)
        """

        raise NotImplementedError