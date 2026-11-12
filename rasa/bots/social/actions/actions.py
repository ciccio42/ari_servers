from itertools import count
from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, AllSlotsReset, FollowupAction
from rasa_sdk.executor import CollectingDispatcher
import logging
from .pluralize import pluralize
import asyncio
from .runner import Runner
import sys
import os
import re 
logger = logging.getLogger(__name__)
# import debugpy
# debugpy.listen(5678)
# print("Waiting client")
# debugpy.wait_for_client()

# from .wiki import Encyclopedia
from .chat_gpt import ChatGpt, DEBUG, FILE_SEARCH_ENABLED
import os

from .text_manager import TextManagerSocial
from .events import NLPEvent
#enc = Encyclopedia()
gpt=ChatGpt()
#gpt.generate_answer("Is this a fake question?") #this API call is because the first call is the slowest

KNOWLEDGE_GRAPH_RUNNING = True if os.environ.get('KNOWLEDGE_GRAPH_RUNNING') == "True" else False 
class ActionGreet(Action):

    def __init__(self):
        super(ActionGreet, self).__init__()
        self._is_first_time = True
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_greet"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        gender = tracker.get_slot('gender')
        age = tracker.get_slot('age')
        

        if age is not None or gender is not None:
            age = int(age)

            if age < 35:
                s = 'young_'
            elif 35 < age < 60:
                s = "midage_"
            else:
                s = "old_"
        
            if gender == 'M':
                s += "man"
            else:
                s += "girl"
            
            saluto = self.TextManager.get_phrase("greetings",s)
        else:
            saluto = self.TextManager.get_phrase("Gr1")
        
        if self._is_first_time:
            hello_sentence = self.TextManager.get_phrase("Gr2").format(saluto)
            self._is_first_time = False
        
        else:
            hello_sentence = saluto
        
        if DEBUG:
            with open("action_greet.txt", "w") as f:
                f.write(hello_sentence)
        dispatcher.utter_message(hello_sentence)

        return [AllSlotsReset()]

class ActionSocial(Action):

    class SocialRunner(Runner):
        def __init__(self, chat_gpt_instance: ChatGpt):
            super().__init__()
            self._chat_gpt_instance = chat_gpt_instance

        def add_task(self, *args, **kwargs):
            return asyncio.run_coroutine_threadsafe(self._chat_gpt_instance.chat_gpt(*args, **kwargs), self.loop)

    def __init__(self):
        super(ActionSocial, self).__init__()
        self._runner = self.SocialRunner(gpt)
        self.language = "Italian"
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_social"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:  

        usr_message = tracker.latest_message['text']
        if DEBUG:
            with open("prompt.txt", "w") as f:
                f.write(usr_message)
        self._runner._chat_gpt_instance.set_prompt(self.TextManager.get_prompt("Medical Prompt"))

        # medical questions: the model also searches the diabetology/cardiology vector store (OPENAI_FILE_SEARCH)
        fut = self._runner.add_task(usr_message, file_search=FILE_SEARCH_ENABLED)
        try:
            definition = fut.result(gpt.TIMEOUT_RESPONSE)
        except Exception as e:
            logger.error(e)
            fut.cancel()
            definition = "" #f"GPT exception: {e}"

        try:
            if definition != "":
                #dispatcher.utter_message(text=f"I know that {definition.lower()} Is it right?")
                dispatcher.utter_message(text=definition)
            else:
                dispatcher.utter_message(text=self.TextManager.get_phrase("So1"))        
        except Exception as e:
            dispatcher.utter_message(text=self.TextManager.get_phrase("So2"))
        return [AllSlotsReset()]

class ActionGPT_4o(Action):
    
    def __init__(self):
        super(ActionGPT_4o, self).__init__()
        self.language = "Italian"
        self.TextManager = TextManagerSocial()
        
    def name(self) -> Text:
        return "action_gpt_4o"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:  
        
        usr_message = tracker.latest_message['text']
        image_base64 = tracker.get_slot("image_slot")
        try:
            if image_base64 is not None:
                response = gpt.gpt_4o(usr_message, image_base64)
            else: 
                logging.error("image slot is empty")
                raise Exception
            if response != "":
                dispatcher.utter_message(text=response)
            else:
                dispatcher.utter_message(text=self.TextManager.get_phrase("So1"))
        except Exception as e:
            dispatcher.utter_message(text=self.TextManager.get_phrase("So2"))
            
        return [AllSlotsReset()]

class ActionDoa(Action):
    def __init__(self):
        super(ActionDoa, self).__init__()
        
    def name(self) -> Text:
        return "action_doa"

    async def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
            ) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text = "Event_Doa: ", buttons = [{"payload": NLPEvent.DOA.value, "title": "DoaEvent"}],)
    
        return [AllSlotsReset()]


class ActionGender(Action):

    def __init__(self):
        super(ActionGender, self).__init__()
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_gender"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        gender = tracker.get_slot('gender')

        if gender is not None:            
            s = self.TextManager.get_phrase("sentences",gender)
            answer = self.TextManager.get_phrase("Ge1").format(s)
        else:
            answer =   self.TextManager.get_phrase("Ge2")
        
        dispatcher.utter_message(text=answer)

        return [AllSlotsReset()]

class ActionEmotion(Action):

    def __init__(self):
        super(ActionEmotion, self).__init__()
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_emotion"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        emotion = tracker.get_slot('emotion')        

        if emotion is not None:

            if emotion == self.TextManager.get_phrase('Em1'):
                sentence = self.TextManager.get_phrase('Em2')
            else:
                #sentence = f"Sembri essere {emotion}."
                sentence=self.TextManager.get_phrase('Em3').format(self.TextManager.get_phrase('em',emotion))
                
        else:
            sentence =  self.TextManager.get_phrase('Em4')

        dispatcher.utter_message(text=sentence)

        return [AllSlotsReset()]

class ActionAge(Action):

    def __init__(self):
        super(ActionAge, self).__init__()
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_age"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        age = tracker.get_slot('age')

        if age is not None:  
            age = int(age)  
            sentence =  self.TextManager.get_phrase('Ag1').format(age)
        else: 
            sentence =  self.TextManager.get_phrase('Ag2')

        dispatcher.utter_message(text=sentence)

        return [AllSlotsReset()]
    
class ActionRockPaperScissor(Action):

    def __init__(self):
        super(ActionRockPaperScissor, self).__init__()
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_start_rock_paper_scissor"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        
        dispatcher.utter_message(text="Event Rock Paper Scissor", buttons = [{"payload": NLPEvent.ROCK_PAPER_SCISSOR.value, "title": "RockPaperScissorEvent"}])

        return [AllSlotsReset()]
    
class ActionConnectFour(Action):
    """
    This class defines a custom Rasa action to start the Connect Four event.
    """
    def __init__(self):
        super(ActionConnectFour, self).__init__()

    def name(self) -> Text:
        """
        Specifies the name of the action.
        Returns:
            str: The name of the action ("action_start_connect_four").
        """
        return "action_start_connect_four"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        """
        Executes the logic of the action. Sends a message to the user with a button 
        to trigger the Connect Four event and resets all slots in the tracker.

        Args:
            dispatcher (CollectingDispatcher): Used to send messages or actions to the user.
            tracker (Tracker): Contains the conversation context, such as the recognized intent 
                and slot values.
            domain (Dict[Text, Any]): A dictionary containing the Rasa domain configuration.

        Returns:
            List[Dict[Text, Any]]: A list of events, in this case resetting all slots
            via the AllSlotsReset event.
        """

        
        dispatcher.utter_message(text="Event Connect Four", buttons = [{"payload": NLPEvent.CONNECT_FOUR.value, "title": "ConnectFourEvent"}])

        return [AllSlotsReset()]

class ActionBiometrics(Action):

    def __init__(self):
        super(ActionBiometrics, self).__init__()
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_biometrics"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        gender = tracker.get_slot('gender')
        g = self.TextManager.get_phrase('Bi1') if gender == 'M' else self.TextManager.get_phrase('Bi2')
        age = tracker.get_slot('age')
        emotion = tracker.get_slot('emotion')

        if emotion is not None:               
            sentence_emotion =  f"{self.TextManager.get_phrase('em',emotion)}."
        
            if emotion == self.TextManager.get_phrase('Bi3'):
                sentence_emotion = self.TextManager.get_phrase('Bi4')

            s = SlotSet("gender",self.TextManager.get_phrase('Bi1') if gender == 'M' else self.TextManager.get_phrase('Bi2'))

            if (g=='uomo'):
              
              dispatcher.utter_message(text=self.TextManager.get_phrase('Bi5').format(g,sentence_emotion,age))
            else:            
              dispatcher.utter_message(text=self.TextManager.get_phrase('Bi6').format(g,sentence_emotion,age))
        else:
            dispatcher.utter_message(text=self.TextManager.get_phrase('Bi7'))

        return [AllSlotsReset()]

class ActionObjects(Action):

    def __init__(self):
        super(ActionObjects, self).__init__()
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_objects"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        objects_list = tracker.get_slot('objects_list')   
        if objects_list is None:
            objects_list = []
        object_type = tracker.get_slot('object_type')

        objects_list = [self.TextManager.get_phrase("coco_labels", objects_list[i]) for i in range(len(objects_list))]
        
        try:
            object_type = self.TextManager.get_phrase("coco_labels", object_type)
        except:
            pass # Ussuming that if is not in coco labels the word is already in italian.

        counter = {}
        if objects_list is not None :
            for obj in objects_list:
                counter[obj] = objects_list.count(obj)
        
        if object_type is not None:
            if object_type not in counter:
                answer = self.TextManager.get_phrase('Ob1')
            else:
                word = pluralize(object_type) if counter[object_type] > 1 else object_type
                answer = self.TextManager.get_phrase('Ob2').format(counter[object_type],word)
        
        else:
            if objects_list is None or len(objects_list) == 0:
                answer = self.TextManager.get_phrase('Ob3')
            else:
                answer = self.TextManager.get_phrase('Ob4')
                for x in counter:
                    word = pluralize(x) if counter[x] > 1 else x
                    answer += f"{counter[x]} {word}, "

        dispatcher.utter_message(text=answer)

        return [AllSlotsReset()]


class ActionAddInfoKG(Action):

    class SocialRunner(Runner):
        def __init__(self, chat_gpt_instance: ChatGpt):
            super().__init__()
            self._chat_gpt_instance = chat_gpt_instance
        
        def add_task(self, *args, **kwargs):
            return asyncio.run_coroutine_threadsafe(self._chat_gpt_instance.chat_gpt(*args, **kwargs), self.loop)

    def __init__(self):
        super(ActionAddInfoKG, self).__init__()
        self._runner = self.SocialRunner(gpt)
        self.language = "Italian"
        self.TextManager = TextManagerSocial()

    def name(self) -> Text:
        return "action_add_info_kg"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:  
        
        if KNOWLEDGE_GRAPH_RUNNING == False:
            return [FollowupAction('action_social')]
        
        usr_message = tracker.latest_message['text']

        if DEBUG:
            with open("prompt_personal_request.txt", "w") as f:
                f.write(usr_message)

        self._runner._chat_gpt_instance.set_prompt(self.TextManager.get_phrase("KG_Prompt"))    
        fut = self._runner.add_task(usr_message)
        try:
            definition = fut.result(gpt.TIMEOUT_RESPONSE)
        except Exception as e:
            logger.error(e)
            fut.cancel()
            definition = "" #f"GPT exception: {e}"

        try:
            if definition != "":
                #dispatcher.utter_message(text=f"I know that {definition.lower()} Is it right?")
                dispatcher.utter_message(text=definition)
            else:
                dispatcher.utter_message(text=self.TextManager.get_phrase("So1"))        
        except Exception as e:
            dispatcher.utter_message(text=self.TextManager.get_phrase("So2"))
        return [AllSlotsReset()]


# class ActionAskRag(Action):

#     class SocialRunner(Runner):
#         def __init__(self):
#             raise NotImplementedError("")
        
#         def add_task(self, *args, **kwargs):
#             raise NotImplementedError("")

#     def __init__(self):
#         raise NotImplementedError("")

#     def name(self) -> Text:
#         return "action_ask_rag"

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:  
        
#         raise NotImplementedError("")


# class ActionSplitDecision(Action):

#     def __init__(self):
#         super(ActionSplitDecision, self).__init__()

#     def name(self) -> Text:
#         return "ActionSplitDecision"
    
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         if KNOWLEDGE_GRAPH_RUNNING:
#             return [FollowupAction('action_add_info_kg')]
#         else:
#             return [FollowupAction('action_social')]



# class ActionIdentity(Action):

#     def __init__(self):
#         super(ActionIdentity, self).__init__()
#         self.TextManager = TextManager(os.environ['LANGUAGE'], "actions/text_manager/phrases_rasa.yml")

#     def name(self) -> Text:
#         return "action_identity"
    
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
#         identity = tracker.get_slot('identity')

#         if identity is None:
#             dispatcher.utter_message(text=self.TextManager.get_phrase('Id1'))
#         elif identity == 'Unknown' or identity == '?':
#             dispatcher.utter_message(text=self.TextManager.get_phrase('Id2'))
#         else:
#             dispatcher.utter_message(text=self.TextManager.get_phrase('Id3').format(identity))

#         return [AllSlotsReset()]
