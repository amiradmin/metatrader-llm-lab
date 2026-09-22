#property strict
#property version   "0.02"
#property description "Lesson 01 - send a market snapshot to the Python bridge"

input string BridgeUrl = "http://127.0.0.1:8010/snapshot";
input int    TimerSeconds = 15;

string TimeframeName()
{
   return EnumToString((ENUM_TIMEFRAMES)_Period);
}

string JsonEscape(const string value)
{
   string out = value;
   StringReplace(out, "\\", "\\\\");
   StringReplace(out, "\"", "\\\"");
   return out;
}

void SendSnapshot()
{
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
   {
      Print("HECTOR LLM LAB | SymbolInfoTick failed");
      return;
   }

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double spread_points = 0.0;
   if(point > 0.0)
      spread_points = (tick.ask - tick.bid) / point;

   string position = PositionSelect(_Symbol) ? "OPEN" : "NONE";
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   string timestamp = StringFormat(
      "%04d-%02d-%02dT%02d:%02d:%02d",
      dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec
   );

   string body = StringFormat(
      "{\"symbol\":\"%s\",\"timeframe\":\"%s\",\"bid\":%.10f,\"ask\":%.10f,\"spread_points\":%.2f,\"position\":\"%s\",\"timestamp\":\"%s\"}",
      JsonEscape(_Symbol),
      JsonEscape(TimeframeName()),
      tick.bid,
      tick.ask,
      spread_points,
      position,
      timestamp
   );

   char request[];
   char response[];
   string response_headers;
   StringToCharArray(body, request, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(request, ArraySize(request)-1);

   string headers = "Content-Type: application/json\r\n";
   ResetLastError();
   int status = WebRequest("POST", BridgeUrl, headers, 5000, request, response, response_headers);

   if(status == -1)
   {
      Print("HECTOR LLM LAB | WebRequest failed | error=", GetLastError(),
            " | Add http://127.0.0.1:8010 to Tools > Options > Expert Advisors > Allow WebRequest");
      return;
   }

   string response_text = CharArrayToString(response, 0, -1, CP_UTF8);
   Print("HECTOR LLM LAB | HTTP=", status,
         " | symbol=", _Symbol,
         " | bid=", DoubleToString(tick.bid, _Digits),
         " | ask=", DoubleToString(tick.ask, _Digits),
         " | spread=", DoubleToString(spread_points, 1),
         " | response=", response_text);
}

int OnInit()
{
   EventSetTimer(MathMax(1, TimerSeconds));
   Print("HECTOR LLM LAB | Lesson 01 started | bridge=", BridgeUrl);
   SendSnapshot();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTimer()
{
   SendSnapshot();
}
