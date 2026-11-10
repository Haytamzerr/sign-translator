/**
 * Lesson topics for the 250 built-in ASL signs. Every sign appears in exactly
 * one topic (a test checks this against the model's label list).
 */

export interface Topic {
  id: string;
  /** Interface text key suffix: t(`topic_${id}`). */
  signs: string[];
}

export const TOPICS: Topic[] = [
  { id: 'greetings', signs: ['hello', 'bye', 'please', 'thankyou', 'yes', 'no', 'fine', 'smile', 'kiss', 'shhh', 'owie'] },
  {
    id: 'people',
    signs: ['mom', 'dad', 'brother', 'aunt', 'uncle', 'grandma', 'grandpa', 'boy', 'girl', 'child', 'man', 'person',
      'fireman', 'police', 'clown', 'cowboy', 'hesheit', 'minemy', 'weus', 'yourself'],
  },
  {
    id: 'animals',
    signs: ['animal', 'alligator', 'bee', 'bird', 'bug', 'cat', 'kitty', 'cow', 'dog', 'puppy', 'donkey', 'duck', 'elephant',
      'fish', 'frog', 'giraffe', 'goose', 'hen', 'horse', 'lion', 'mouse', 'owl', 'pig', 'tiger', 'wolf', 'zebra'],
  },
  {
    id: 'food',
    signs: ['food', 'drink', 'water', 'milk', 'apple', 'carrot', 'cereal', 'chocolate', 'frenchfries', 'gum', 'icecream',
      'nuts', 'pizza', 'snack', 'taste', 'hungry', 'thirsty', 'yucky'],
  },
  { id: 'colors', signs: ['black', 'blue', 'brown', 'green', 'orange', 'red', 'white', 'yellow'] },
  {
    id: 'body',
    signs: ['head', 'face', 'hair', 'eye', 'ear', 'nose', 'mouth', 'lips', 'tongue', 'tooth', 'cheek', 'chin', 'arm', 'finger', 'feet'],
  },
  { id: 'clothes', signs: ['shirt', 'jacket', 'jeans', 'hat', 'mitten', 'pajamas', 'shoe', 'underwear', 'zipper'] },
  {
    id: 'home',
    signs: ['home', 'room', 'bedroom', 'bed', 'bath', 'shower', 'potty', 'stairs', 'backyard', 'pool', 'chair', 'table', 'lamp',
      'closet', 'drawer', 'refrigerator', 'dryer', 'vacuum', 'glasswindow', 'TV', 'radio', 'book', 'pen', 'pencil',
      'scissors', 'napkin', 'toothbrush', 'garbage', 'flag', 'penny', 'gift', 'toy', 'doll', 'puzzle', 'balloon'],
  },
  {
    id: 'nature',
    signs: ['outside', 'farm', 'tree', 'flower', 'grass', 'sun', 'moon', 'cloud', 'rain', 'snow', 'airplane', 'boat', 'car',
      'helicopter', 'store'],
  },
  {
    id: 'feelings',
    signs: ['happy', 'sad', 'mad', 'cry', 'sick', 'sleepy', 'awake', 'hate', 'like', 'bad', 'better', 'pretty', 'cute', 'old',
      'same', 'clean', 'dirty', 'wet', 'dry', 'hot', 'sticky', 'stuck', 'empty', 'fast', 'loud', 'noisy', 'quiet', 'high'],
  },
  {
    id: 'actions',
    signs: ['go', 'stay', 'wait', 'look', 'see', 'listen', 'hear', 'talk', 'say', 'read', 'think', 'find', 'hide', 'give',
      'have', 'haveto', 'make', 'can', 'will', 'finish', 'open', 'close', 'cut', 'drop', 'fall', 'jump', 'dance', 'ride',
      'blow', 'touch', 'pretend', 'sleep', 'nap', 'wake', 'callonphone', 'story'],
  },
  { id: 'time', signs: ['now', 'later', 'before', 'after', 'first', 'time', 'morning', 'night', 'yesterday', 'tomorrow'] },
  {
    id: 'questions',
    signs: ['who', 'where', 'why', 'if', 'because', 'for', 'that', 'there', 'into', 'on', 'up', 'down', 'beside', 'all', 'any',
      'another', 'every', 'many', 'not'],
  },
];

/** The ASL manual alphabet is its own lesson. */
export const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
